/* Python bindings for rocm-xio SDMA endpoint (host-side only).
 *
 * Provides the xio.sdma_ep module that iris imports:
 *   from xio import sdma_ep
 *
 * Wraps the existing rocm-xio C++ host-side API directly.
 * Links against librocm-xio.so — no source stripping or patching.
 *
 * Host-side put/signal/quiet are implemented here because the
 * existing C++ code only has __device__ versions. These write
 * SDMA packets to the ring buffer from CPU and ring the doorbell.
 * The queue buffer has HostAccess=1 (set by anvil's KFD alloc).
 */

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <hip/hip_runtime.h>

#include "anvil.hpp"
#include "sdma-ep.h"
#include "sdma_pkt_struct.h"
#include "sdma_opcodes.h"

namespace py = pybind11;

// ---- Constants matching iris's expectations ----
static constexpr int QUEUE_DEVICE_CTX_SIZE = 6;
static constexpr int COPY_LINEAR_COMMAND_BYTES = 7 * 4;
static constexpr int COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES = 20 * 4;
static constexpr int ATOMIC_COMMAND_BYTES = 8 * 4;
static constexpr int POLL_REGMEM_COMMAND_BYTES = 6 * 4;

// ---- Tile class matching iris's sdma_ep.Tile ----
struct Tile {
  uint64_t data_ptr = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t stride = 0;
};

// ---- Track created queues ----
struct QueueEntry {
  anvil::SdmaQueue* queue;
  anvil::SdmaQueueDeviceHandle* deviceHandle;
};

static std::unordered_map<uint64_t, QueueEntry> s_queues;

static uint64_t make_key(int src, int dst) {
  return (static_cast<uint64_t>(src) << 32) | static_cast<uint64_t>(dst);
}

// ---- Host-side SDMA packet helpers ----
// wptr/rptr/doorbell are KFD memory-mapped pointers, CPU-accessible.
// queueBuf is allocated with HostAccess=1 via hsaKmtAllocMemory.

static uint64_t read_wptr(anvil::SdmaQueueDeviceHandle* h) {
  return *reinterpret_cast<volatile uint64_t*>(h->wptr);
}

static void advance_and_ring(anvil::SdmaQueueDeviceHandle* h,
                             uint64_t new_wptr) {
  *reinterpret_cast<volatile uint64_t*>(h->wptr) = new_wptr;
  *reinterpret_cast<volatile uint64_t*>(h->doorbell) = new_wptr;
}

static uint32_t* slot_at(anvil::SdmaQueueDeviceHandle* h, uint64_t wptr) {
  uint64_t offset = wptr % anvil::SDMA_QUEUE_SIZE;
  return reinterpret_cast<uint32_t*>(
    reinterpret_cast<uint8_t*>(h->queueBuf) + offset);
}

static void host_write_copy_linear(anvil::SdmaQueueDeviceHandle* h,
                                   uint64_t src_addr, uint64_t dst_addr,
                                   uint64_t size) {
  uint64_t wptr = read_wptr(h);
  uint32_t* s = slot_at(h, wptr);

  uint32_t pkt[7] = {};
  pkt[0] = (SDMA_SUBOP_COPY_LINEAR << 8) | SDMA_OP_COPY;
  pkt[1] = static_cast<uint32_t>(size - 1);
  pkt[2] = 0;
  pkt[3] = static_cast<uint32_t>(src_addr);
  pkt[4] = static_cast<uint32_t>(src_addr >> 32);
  pkt[5] = static_cast<uint32_t>(dst_addr);
  pkt[6] = static_cast<uint32_t>(dst_addr >> 32);

  std::memcpy(s, pkt, sizeof(pkt));
  advance_and_ring(h, wptr + sizeof(pkt));
}

static void host_write_atomic_inc(anvil::SdmaQueueDeviceHandle* h,
                                  uint64_t signal_addr) {
  uint64_t wptr = read_wptr(h);
  uint32_t* s = slot_at(h, wptr);

  uint32_t pkt[8] = {};
  pkt[0] = (SDMA_ATOMIC_ADD64 << 25) | SDMA_OP_ATOMIC;
  pkt[1] = static_cast<uint32_t>(signal_addr);
  pkt[2] = static_cast<uint32_t>(signal_addr >> 32);
  pkt[3] = 1;
  pkt[4] = 0;
  pkt[5] = 0;
  pkt[6] = 0;
  pkt[7] = 0;

  std::memcpy(s, pkt, sizeof(pkt));
  advance_and_ring(h, wptr + sizeof(pkt));
}

static void host_write_poll_regmem(anvil::SdmaQueueDeviceHandle* h,
                                   uint64_t addr, uint64_t ref,
                                   uint64_t mask, uint32_t func) {
  uint64_t wptr = read_wptr(h);
  uint32_t* s = slot_at(h, wptr);

  uint32_t pkt[6] = {};
  pkt[0] = SDMA_OP_POLL_REGMEM | (func << 28) | (1u << 31);
  pkt[1] = static_cast<uint32_t>(addr);
  pkt[2] = static_cast<uint32_t>(addr >> 32);
  pkt[3] = static_cast<uint32_t>(ref);
  pkt[4] = static_cast<uint32_t>(mask);
  pkt[5] = 0;

  std::memcpy(s, pkt, sizeof(pkt));
  advance_and_ring(h, wptr + sizeof(pkt));
}

static void host_write_sub_window_copy(anvil::SdmaQueueDeviceHandle* h,
                                       uint64_t src_addr, uint64_t dst_addr,
                                       uint32_t tile_width, uint32_t tile_height,
                                       uint32_t src_pitch, uint32_t dst_pitch,
                                       uint32_t src_x, uint32_t src_y,
                                       uint32_t dst_x, uint32_t dst_y) {
  uint64_t wptr = read_wptr(h);
  uint32_t* s = slot_at(h, wptr);

  uint32_t pkt[20] = {};
  pkt[0] = (SDMA_SUBOP_COPY_LINEAR_SUB_WINDOW << 8) | SDMA_OP_COPY;
  pkt[1] = static_cast<uint32_t>(src_addr);
  pkt[2] = static_cast<uint32_t>(src_addr >> 32);
  pkt[3] = src_x;
  pkt[4] = src_y;
  pkt[5] = 0;
  pkt[6] = src_pitch - 1;
  pkt[7] = 0;
  pkt[8] = 0;
  pkt[9] = static_cast<uint32_t>(dst_addr);
  pkt[10] = static_cast<uint32_t>(dst_addr >> 32);
  pkt[11] = dst_x;
  pkt[12] = dst_y;
  pkt[13] = 0;
  pkt[14] = dst_pitch - 1;
  pkt[15] = 0;
  pkt[16] = 0;
  pkt[17] = tile_width - 1;
  pkt[18] = tile_height - 1;
  pkt[19] = 0;

  std::memcpy(s, pkt, sizeof(pkt));
  advance_and_ring(h, wptr + sizeof(pkt));
}

static void host_quiet(anvil::SdmaQueueDeviceHandle* h) {
  uint64_t wptr = read_wptr(h);
  int retries = 0;
  while (true) {
    uint64_t rptr = *reinterpret_cast<volatile uint64_t*>(h->rptr);
    if (rptr >= wptr) break;
    if (++retries > (1 << 30))
      throw std::runtime_error("quiet: SDMA engine timed out");
  }
}

// ---- Python module ----

PYBIND11_MODULE(sdma_ep, m) {
  m.doc() = "rocm-xio SDMA endpoint Python bindings (host-side)";

  // Constants
  m.attr("SDMA_QUEUE_SIZE") = static_cast<int>(anvil::SDMA_QUEUE_SIZE);
  m.attr("QUEUE_DEVICE_CTX_SIZE") = QUEUE_DEVICE_CTX_SIZE;
  m.attr("COPY_LINEAR_COMMAND_BYTES") = COPY_LINEAR_COMMAND_BYTES;
  m.attr("COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES") = COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES;
  m.attr("ATOMIC_COMMAND_BYTES") = ATOMIC_COMMAND_BYTES;
  m.attr("POLL_REGMEM_COMMAND_BYTES") = POLL_REGMEM_COMMAND_BYTES;

  // Tile class
  py::class_<Tile>(m, "Tile")
    .def(py::init<>())
    .def_readwrite("data_ptr", &Tile::data_ptr)
    .def_readwrite("width", &Tile::width)
    .def_readwrite("height", &Tile::height)
    .def_readwrite("stride", &Tile::stride);

  // init() — wraps xio::sdma_ep::initEndpoint
  m.def("init", []() {
    int rc = xio::sdma_ep::initEndpoint();
    if (rc != 0) throw std::runtime_error("sdma_ep.init() failed");
  });

  // create_queue(src_rank, dst_rank)
  m.def("create_queue", [](int src_rank, int dst_rank) {
    uint64_t key = make_key(src_rank, dst_rank);
    if (s_queues.count(key)) return;

    xio::sdma_ep::SdmaConnectionInfo conn;
    int rc = xio::sdma_ep::createConnection(src_rank, dst_rank, &conn);
    if (rc != 0) throw std::runtime_error("createConnection failed");

    xio::sdma_ep::SdmaQueueInfo info;
    rc = xio::sdma_ep::createQueue(src_rank, dst_rank, &info);
    if (rc != 0) throw std::runtime_error("createQueue failed");

    auto* q = anvil::AnvilLib::getInstance().getSdmaQueue(
      src_rank, dst_rank, info.channelIdx);
    s_queues[key] = {q, static_cast<anvil::SdmaQueueDeviceHandle*>(
                          info.deviceHandle)};
  });

  // create_host_queue — same path
  m.def("create_host_queue", [](int src_rank, int dst_rank) {
    uint64_t key = make_key(src_rank, dst_rank);
    if (s_queues.count(key)) return;

    xio::sdma_ep::SdmaConnectionInfo conn;
    int rc = xio::sdma_ep::createConnection(src_rank, dst_rank, &conn);
    if (rc != 0) throw std::runtime_error("createConnection failed");

    xio::sdma_ep::SdmaQueueInfo info;
    rc = xio::sdma_ep::createQueue(src_rank, dst_rank, &info);
    if (rc != 0) throw std::runtime_error("createQueue failed");

    auto* q = anvil::AnvilLib::getInstance().getSdmaQueue(
      src_rank, dst_rank, info.channelIdx);
    s_queues[key] = {q, static_cast<anvil::SdmaQueueDeviceHandle*>(
                          info.deviceHandle)};
  });

  // get_queue_device_ctx(src_rank, dst_rank)
  m.def("get_queue_device_ctx", [](int src_rank, int dst_rank) -> py::object {
    uint64_t key = make_key(src_rank, dst_rank);
    auto it = s_queues.find(key);
    if (it == s_queues.end())
      throw std::runtime_error("Queue not created for this rank pair");

    auto* h = it->second.deviceHandle;
    py::object ns = py::module_::import("types").attr("SimpleNamespace");
    return ns(
      py::arg("queue_buf") = reinterpret_cast<uint64_t>(h->queueBuf),
      py::arg("rptr") = reinterpret_cast<uint64_t>(h->rptr),
      py::arg("wptr") = reinterpret_cast<uint64_t>(h->wptr),
      py::arg("doorbell") = reinterpret_cast<uint64_t>(h->doorbell),
      py::arg("cached_wptr") = reinterpret_cast<uint64_t>(h->cachedWptr),
      py::arg("committed_wptr") = reinterpret_cast<uint64_t>(h->committedWptr)
    );
  });

  // Host-side put/signal/quiet — writes SDMA packets from CPU
  m.def("put", [](int src_rank, int dst_rank, int channel,
                   uint64_t src_ptr, uint64_t dst_ptr, uint64_t size) {
    (void)channel;
    auto it = s_queues.find(make_key(src_rank, dst_rank));
    if (it == s_queues.end()) throw std::runtime_error("Queue not found");
    host_write_copy_linear(it->second.deviceHandle, src_ptr, dst_ptr, size);
  });

  m.def("put_signal", [](int src_rank, int dst_rank, int channel,
                          uint64_t src_ptr, uint64_t dst_ptr, uint64_t size,
                          uint64_t signal_ptr, uint64_t signal_val,
                          int signal_bits) {
    (void)channel; (void)signal_val; (void)signal_bits;
    auto it = s_queues.find(make_key(src_rank, dst_rank));
    if (it == s_queues.end()) throw std::runtime_error("Queue not found");
    auto* h = it->second.deviceHandle;
    host_write_copy_linear(h, src_ptr, dst_ptr, size);
    host_write_atomic_inc(h, signal_ptr);
  });

  m.def("signal", [](int src_rank, int dst_rank, int channel,
                      uint64_t signal_ptr, uint64_t signal_val,
                      int signal_bits) {
    (void)channel; (void)signal_val; (void)signal_bits;
    auto it = s_queues.find(make_key(src_rank, dst_rank));
    if (it == s_queues.end()) throw std::runtime_error("Queue not found");
    host_write_atomic_inc(it->second.deviceHandle, signal_ptr);
  });

  m.def("quiet", [](int src_rank, int dst_rank, int channel) {
    (void)channel;
    auto it = s_queues.find(make_key(src_rank, dst_rank));
    if (it == s_queues.end()) throw std::runtime_error("Queue not found");
    host_quiet(it->second.deviceHandle);
  });

  m.def("put_tile", [](int src_rank, int dst_rank, int channel,
                        const Tile& tile, uint64_t dst_ptr,
                        uint64_t dst_stride) {
    (void)channel;
    auto it = s_queues.find(make_key(src_rank, dst_rank));
    if (it == s_queues.end()) throw std::runtime_error("Queue not found");
    host_write_sub_window_copy(
      it->second.deviceHandle, tile.data_ptr, dst_ptr,
      tile.width, tile.height, tile.stride,
      static_cast<uint32_t>(dst_stride), 0, 0, 0, 0);
  });

  m.def("put_tile_signal", [](int src_rank, int dst_rank, int channel,
                               const Tile& tile, uint64_t dst_ptr,
                               uint64_t dst_stride, uint64_t signal_ptr,
                               uint64_t signal_val, int signal_bits) {
    (void)channel; (void)signal_val; (void)signal_bits;
    auto it = s_queues.find(make_key(src_rank, dst_rank));
    if (it == s_queues.end()) throw std::runtime_error("Queue not found");
    auto* h = it->second.deviceHandle;
    host_write_sub_window_copy(
      h, tile.data_ptr, dst_ptr,
      tile.width, tile.height, tile.stride,
      static_cast<uint32_t>(dst_stride), 0, 0, 0, 0);
    host_write_atomic_inc(h, signal_ptr);
  });

  m.def("wait_flag_then_put", [](int src_rank, int dst_rank, int channel,
                                  uint64_t src_ptr, uint64_t dst_ptr,
                                  uint64_t size, uint64_t flag_ptr,
                                  uint64_t flag_val, int flag_bits) {
    (void)channel; (void)flag_bits;
    auto it = s_queues.find(make_key(src_rank, dst_rank));
    if (it == s_queues.end()) throw std::runtime_error("Queue not found");
    auto* h = it->second.deviceHandle;
    host_write_poll_regmem(h, flag_ptr, flag_val, 0xFFFFFFFF, 5);
    host_write_copy_linear(h, src_ptr, dst_ptr, size);
  });

  m.def("wait_flag_then_put_tile", [](int src_rank, int dst_rank, int channel,
                                       const Tile& tile, uint64_t dst_ptr,
                                       uint64_t dst_stride, uint64_t flag_ptr,
                                       uint64_t flag_val, int flag_bits) {
    (void)channel; (void)flag_bits;
    auto it = s_queues.find(make_key(src_rank, dst_rank));
    if (it == s_queues.end()) throw std::runtime_error("Queue not found");
    auto* h = it->second.deviceHandle;
    host_write_poll_regmem(h, flag_ptr, flag_val, 0xFFFFFFFF, 5);
    host_write_sub_window_copy(
      h, tile.data_ptr, dst_ptr,
      tile.width, tile.height, tile.stride,
      static_cast<uint32_t>(dst_stride), 0, 0, 0, 0);
  });

  m.def("put_tiles", [](int src_rank, int dst_rank, int channel,
                         const std::vector<Tile>& tiles,
                         const std::vector<uint64_t>& dst_ptrs,
                         const std::vector<uint64_t>& dst_strides) {
    (void)channel;
    auto it = s_queues.find(make_key(src_rank, dst_rank));
    if (it == s_queues.end()) throw std::runtime_error("Queue not found");
    auto* h = it->second.deviceHandle;
    for (size_t i = 0; i < tiles.size(); i++) {
      host_write_sub_window_copy(
        h, tiles[i].data_ptr, dst_ptrs[i],
        tiles[i].width, tiles[i].height, tiles[i].stride,
        static_cast<uint32_t>(dst_strides[i]), 0, 0, 0, 0);
    }
  });

  m.def("wait_flag_then_put_tiles", [](int src_rank, int dst_rank, int channel,
                                        const std::vector<Tile>& tiles,
                                        const std::vector<uint64_t>& dst_ptrs,
                                        const std::vector<uint64_t>& dst_strides,
                                        uint64_t flag_ptr, uint64_t flag_val,
                                        int flag_bits) {
    (void)channel; (void)flag_bits;
    auto it = s_queues.find(make_key(src_rank, dst_rank));
    if (it == s_queues.end()) throw std::runtime_error("Queue not found");
    auto* h = it->second.deviceHandle;
    host_write_poll_regmem(h, flag_ptr, flag_val, 0xFFFFFFFF, 5);
    for (size_t i = 0; i < tiles.size(); i++) {
      host_write_sub_window_copy(
        h, tiles[i].data_ptr, dst_ptrs[i],
        tiles[i].width, tiles[i].height, tiles[i].stride,
        static_cast<uint32_t>(dst_strides[i]), 0, 0, 0, 0);
    }
  });
}
