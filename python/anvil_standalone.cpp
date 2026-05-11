/* Standalone anvil implementation — no xio.h dependency. */

#include "anvil_standalone.hpp"

#define CHECK_HSAKMT_SUCCESS(call, msg)                                        \
  do {                                                                         \
    if ((call) != HSAKMT_STATUS_SUCCESS) {                                     \
      std::cerr << "HSAKMT error: " << msg << " at " << __FILE__ << ":"       \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CHECK_HIP_ERROR(cmd)                                                   \
  do {                                                                         \
    hipError_t _e = (cmd);                                                     \
    if (_e != hipSuccess) {                                                    \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << #cmd << ": " << hipGetErrorString(_e) << std::endl;         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

namespace anvil {

// ---- Peer access ----

void EnablePeerAccess(int deviceId, int peerDeviceId) {
  int canAccess;
  CHECK_HIP_ERROR(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
  if (!canAccess) {
    std::cerr << "Cannot enable peer access " << deviceId << " -> "
              << peerDeviceId << std::endl;
    return;
  }
  CHECK_HIP_ERROR(hipSetDevice(deviceId));
  hipError_t err = hipDeviceEnablePeerAccess(peerDeviceId, 0);
  if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
    std::cerr << "Peer access failed: " << hipGetErrorString(err) << std::endl;
  }
}

// ---- HSA agent discovery ----

static std::vector<hsa_agent_t> cpuAgents_;
static std::vector<hsa_agent_t> gpuAgents_;

static hsa_status_t agent_callback(hsa_agent_t agent,
                                   hsa_device_type_t target, void* vec) {
  auto* agents = static_cast<std::vector<hsa_agent_t>*>(vec);
  hsa_device_type_t type{};
  hsa_status_t s = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (s != HSA_STATUS_SUCCESS) return s;
  if (type == target) agents->push_back(agent);
  return s;
}

static hsa_status_t gpu_agent_cb(hsa_agent_t a, void* c) {
  return agent_callback(a, HSA_DEVICE_TYPE_GPU, c);
}
static hsa_status_t cpu_agent_cb(hsa_agent_t a, void* c) {
  return agent_callback(a, HSA_DEVICE_TYPE_CPU, c);
}

static void SetUpKFD() {
  CHECK_HSAKMT_SUCCESS(hsaKmtOpenKFD(), "hsaKmtOpenKFD failed");
  HsaSystemProperties props;
  std::memset(&props, 0, sizeof(props));
  CHECK_HSAKMT_SUCCESS(hsaKmtAcquireSystemProperties(&props),
                       "AcquireSystemProperties failed");
}

static bool s_kfd_opened = false;

// ---- SdmaQueue ----

SdmaQueue::SdmaQueue(int localDeviceId, int remoteDeviceId,
                     hsa_agent_t& localAgent, uint32_t engineId) {
  (void)remoteDeviceId;

  uint32_t localNodeId;
  hsa_agent_get_info(localAgent, HSA_AGENT_INFO_NODE, &localNodeId);

  HsaMemFlags memFlags = {};
  memFlags.ui32.NonPaged = 1;
  memFlags.ui32.HostAccess = 1;
  memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  memFlags.ui32.NoNUMABind = 1;
  memFlags.ui32.ExecuteAccess = 1;
  memFlags.ui32.Uncached = 1;

  CHECK_HSAKMT_SUCCESS(
    hsaKmtAllocMemory(localNodeId, SDMA_QUEUE_SIZE, memFlags, &queueBuffer_),
    "AllocMemory failed");
  CHECK_HSAKMT_SUCCESS(
    hsaKmtMapMemoryToGPU(queueBuffer_, SDMA_QUEUE_SIZE, NULL),
    "MapMemoryToGPU failed");

  std::memset(&queue_, 0, sizeof(HsaQueueResource));
  CHECK_HSAKMT_SUCCESS(
    hsaKmtCreateQueueExt(localNodeId, HSA_QUEUE_SDMA_BY_ENG_ID,
                         DEFAULT_QUEUE_PERCENTAGE, DEFAULT_PRIORITY, engineId,
                         queueBuffer_, SDMA_QUEUE_SIZE, nullptr, &queue_),
    "CreateQueueExt failed");

  // Replace xio::allocDeviceMemory with direct HIP calls
  CHECK_HIP_ERROR(hipMalloc((void**)&deviceHandle_,
                            sizeof(SdmaQueueDeviceHandle)));
  CHECK_HIP_ERROR(hipExtMallocWithFlags((void**)&cachedWptr_,
                                         sizeof(uint64_t),
                                         hipDeviceMallocUncached));
  CHECK_HIP_ERROR(hipExtMallocWithFlags((void**)&committedWptr_,
                                         sizeof(uint64_t),
                                         hipDeviceMallocUncached));

  uint64_t cachedWptr = *reinterpret_cast<uint64_t*>(queue_.Queue_write_ptr_aql);
  uint64_t committedWptr = *reinterpret_cast<uint64_t*>(queue_.Queue_write_ptr_aql);
  SdmaQueueDeviceHandle handle = {
    .queueBuf = static_cast<uint32_t*>(queueBuffer_),
    .rptr = queue_.Queue_read_ptr_aql,
    .wptr = queue_.Queue_write_ptr_aql,
    .doorbell = queue_.Queue_DoorBell_aql,
    .cachedWptr = cachedWptr_,
    .committedWptr = committedWptr_,
    .cachedHwReadIndex = *reinterpret_cast<uint64_t*>(queue_.Queue_read_ptr_aql),
    .maxWritePtr = *reinterpret_cast<uint64_t*>(queue_.Queue_read_ptr_aql),
  };

  CHECK_HIP_ERROR(hipMemcpy(deviceHandle_, &handle,
                            sizeof(SdmaQueueDeviceHandle),
                            hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(cachedWptr_, &cachedWptr, sizeof(uint64_t),
                            hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(committedWptr_, &committedWptr, sizeof(uint64_t),
                            hipMemcpyHostToDevice));
}

SdmaQueue::~SdmaQueue() {
  CHECK_HSAKMT_SUCCESS(hsaKmtDestroyQueue(queue_.QueueId),
                       "DestroyQueue failed");
  hipFree(deviceHandle_);
  hipFree(cachedWptr_);
  hipFree(committedWptr_);
  CHECK_HSAKMT_SUCCESS(hsaKmtUnmapMemoryToGPU(queueBuffer_),
                       "UnmapMemoryToGPU failed");
  CHECK_HSAKMT_SUCCESS(hsaKmtFreeMemory(queueBuffer_, SDMA_QUEUE_SIZE),
                       "FreeMemory failed");
}

// ---- AnvilLib ----

AnvilLib& AnvilLib::getInstance() {
  static AnvilLib instance;
  return instance;
}

AnvilLib::~AnvilLib() {
  for (auto& p : sdma_channels_) p.second.clear();
  if (s_kfd_opened) {
    CHECK_HSAKMT_SUCCESS(hsaKmtCloseKFD(), "CloseKFD failed");
    hsa_shut_down();
  }
}

void AnvilLib::init() {
  std::call_once(init_flag, []() {
    hsa_status_t s = hsa_init();
    if (s != HSA_STATUS_SUCCESS)
      throw std::runtime_error("hsa_init failed");
    hsa_iterate_agents(&gpu_agent_cb, &gpuAgents_);
    hsa_iterate_agents(&cpu_agent_cb, &cpuAgents_);
    SetUpKFD();
    s_kfd_opened = true;
  });
}

SdmaQueue* AnvilLib::createSdmaQueue(int srcDeviceId, int dstDeviceId,
                                     uint32_t engineId, int* channelIdx) {
  auto& vec = sdma_channels_[dstDeviceId];
  vec.emplace_back(std::make_unique<SdmaQueue>(
    srcDeviceId, dstDeviceId, gpuAgents_[srcDeviceId], engineId));
  if (channelIdx) *channelIdx = static_cast<int>(vec.size() - 1);
  return vec.back().get();
}

SdmaQueue* AnvilLib::getSdmaQueue(int srcDeviceId, int dstDeviceId,
                                  int channelIdx) {
  (void)srcDeviceId;
  auto it = sdma_channels_.find(dstDeviceId);
  if (it == sdma_channels_.end()) return nullptr;
  if (channelIdx < 0 || static_cast<size_t>(channelIdx) >= it->second.size())
    return nullptr;
  return it->second[channelIdx].get();
}

int AnvilLib::getSdmaEngineId(int srcDeviceId, int dstDeviceId) {
  int srcOamId = getOamId(srcDeviceId);
  int dstOamId = getOamId(dstDeviceId);
  return mi300xOamMap[srcOamId][dstOamId] * 2;
}

int AnvilLib::getOamId(int deviceId) {
  char busIdChar[] = "00000000:00:00.0";
  CHECK_HIP_ERROR(hipDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), deviceId));
  for (size_t i = 0; i < sizeof(busIdChar); i++)
    busIdChar[i] = std::tolower(busIdChar[i]);
  std::string file_str = "/sys/bus/pci/devices/" + std::string(busIdChar) +
                         "/xgmi_physical_id";
  std::ifstream file(file_str);
  int id;
  if (!file.is_open() || !(file >> id))
    throw std::runtime_error("Failed to read xGMI physical id: " + file_str);
  return id;
}

} // namespace anvil
