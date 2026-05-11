/* Standalone anvil host-side API for Python bindings.
 *
 * No xio.h dependency — uses direct HIP/HSA/hsakmt calls.
 * Only needs standard ROCm SDK at build and runtime.
 */

#pragma once

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsakmt/hsakmt.h>
#include <hsakmt/hsakmttypes.h>

namespace anvil {

constexpr uint64_t SDMA_QUEUE_SIZE = 1024 * 1024;
constexpr HSA_QUEUE_PRIORITY DEFAULT_PRIORITY = HSA_QUEUE_PRIORITY_NORMAL;
constexpr unsigned int DEFAULT_QUEUE_PERCENTAGE = 100;

// SdmaQueueDeviceHandle — must match the struct in sdma-ep.h
struct SdmaQueueDeviceHandle {
  uint32_t* queueBuf;
  uint64_t* rptr;
  uint64_t* wptr;
  uint64_t* doorbell;
  uint64_t* cachedWptr;
  uint64_t* committedWptr;
  uint64_t cachedHwReadIndex;
  uint64_t maxWritePtr;
};

void EnablePeerAccess(int deviceId, int peerDeviceId);

class SdmaQueue {
public:
  SdmaQueue(int localDeviceId, int remoteDeviceId,
            hsa_agent_t& localAgent, uint32_t engineId);
  ~SdmaQueue();
  SdmaQueueDeviceHandle* deviceHandle() const { return deviceHandle_; }

private:
  uint64_t* cachedWptr_;
  uint64_t* committedWptr_;
  void* queueBuffer_;
  HsaQueueResource queue_;
  SdmaQueueDeviceHandle* deviceHandle_;
};

class AnvilLib {
public:
  static AnvilLib& getInstance();
  ~AnvilLib();

  void init();
  SdmaQueue* createSdmaQueue(int srcDeviceId, int dstDeviceId,
                              uint32_t engineId, int* channelIdx = nullptr);
  SdmaQueue* getSdmaQueue(int srcDeviceId, int dstDeviceId,
                           int channelIdx = 0);
  int getSdmaEngineId(int srcDeviceId, int dstDeviceId);

private:
  AnvilLib() = default;
  AnvilLib(const AnvilLib&) = delete;
  AnvilLib& operator=(const AnvilLib&) = delete;

  int getOamId(int deviceId);

  std::array<std::array<int, 8>, 8> mi300xOamMap = {{
    {0, 7, 6, 1, 2, 4, 5, 3},
    {7, 0, 1, 5, 4, 2, 3, 6},
    {5, 1, 0, 6, 7, 3, 2, 4},
    {1, 6, 5, 0, 3, 7, 4, 2},
    {2, 4, 7, 3, 0, 5, 6, 1},
    {4, 2, 3, 7, 6, 0, 1, 5},
    {5, 3, 2, 4, 6, 1, 0, 7},
    {3, 6, 4, 2, 1, 5, 7, 0},
  }};

  std::once_flag init_flag;
  std::unordered_map<int, std::vector<std::unique_ptr<SdmaQueue>>> sdma_channels_;
};

} // namespace anvil
