#!/usr/bin/env python3
"""Tests for xio.sdma_ep Python bindings.

Usage:
  python test_sdma_ep.py          # import + constants only
  python test_sdma_ep.py --full   # requires 2 GPUs
"""
import argparse
import sys


def test_import():
    from xio import sdma_ep
    print(f"SDMA_QUEUE_SIZE = {sdma_ep.SDMA_QUEUE_SIZE}")
    print(f"QUEUE_DEVICE_CTX_SIZE = {sdma_ep.QUEUE_DEVICE_CTX_SIZE}")
    print(f"COPY_LINEAR_COMMAND_BYTES = {sdma_ep.COPY_LINEAR_COMMAND_BYTES}")
    print(f"COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES = {sdma_ep.COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES}")
    print(f"ATOMIC_COMMAND_BYTES = {sdma_ep.ATOMIC_COMMAND_BYTES}")
    print(f"POLL_REGMEM_COMMAND_BYTES = {sdma_ep.POLL_REGMEM_COMMAND_BYTES}")

    t = sdma_ep.Tile()
    t.data_ptr = 0xDEAD
    t.width = 64
    t.height = 32
    t.stride = 128
    assert t.data_ptr == 0xDEAD
    assert t.width == 64
    print("import + constants: PASS")


def test_queue_setup():
    from xio import sdma_ep
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"SKIP queue setup: need 2 GPUs, have {n_gpus}")
        return

    sdma_ep.init()
    print("init: OK")

    sdma_ep.create_queue(0, 1)
    print("create_queue(0, 1): OK")

    ctx = sdma_ep.get_queue_device_ctx(0, 1)
    print(f"queue_buf = 0x{ctx.queue_buf:x}")
    print(f"rptr      = 0x{ctx.rptr:x}")
    print(f"wptr      = 0x{ctx.wptr:x}")
    print(f"doorbell  = 0x{ctx.doorbell:x}")
    print("queue setup: PASS")


def test_data_transfer():
    from xio import sdma_ep
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"SKIP data transfer: need 2 GPUs, have {n_gpus}")
        return

    sdma_ep.init()
    sdma_ep.create_queue(0, 1)

    size = 1024
    src = torch.ones(size, dtype=torch.float32, device="cuda:0")
    dst = torch.zeros(size, dtype=torch.float32, device="cuda:1")

    sdma_ep.put(0, 1, 0, src.data_ptr(), dst.data_ptr(), size * 4)
    sdma_ep.quiet(0, 1, 0)

    dst_cpu = dst.cpu()
    expected = torch.ones(size, dtype=torch.float32)
    if torch.allclose(dst_cpu, expected):
        print("data transfer: PASS")
    else:
        diff = (dst_cpu - expected).abs().max().item()
        print(f"data transfer: FAIL (max diff = {diff})")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run full tests (requires 2 GPUs)")
    args = parser.parse_args()

    test_import()
    if args.full:
        test_queue_setup()
        test_data_transfer()
