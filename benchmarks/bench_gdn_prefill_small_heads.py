"""
Benchmark FlashInfer vs FLA/Triton GDN prefill kernel for small-head (GVA) configs.

Demonstrates that FlashInfer Blackwell CuTe-DSL kernel underperforms when
h_v is small (e.g. TP4 of Qwen3.5-397B), because the kernel parallelizes
only over heads — not over sequence chunks — leading to low SM utilization.

Usage:
    python benchmarks/bench_gdn_prefill_small_heads.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from flashinfer.testing import bench_gpu_time
from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_gdn
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_gdn

CONFIGS = [
    # (batch, seq_len, h_qk, h_v, d, label)
    # Qwen3.5-397B (h_k=16, h_v=64, d=128) under different TP
    (1, 8192, 4, 16, 128, "TP4  B=1  S=8192"),
    (1, 4096, 4, 16, 128, "TP4  B=1  S=4096"),
    (1, 2048, 4, 16, 128, "TP4  B=1  S=2048"),
    (1, 8192, 8, 32, 128, "TP2  B=1  S=8192"),
    (1, 8192, 16, 64, 128, "TP1  B=1  S=8192"),
    # Original benchmark config (symmetric heads)
    (4, 4096, 32, 32, 128, "Symmetric B=4 S=4096"),
]

WARMUP = 5
ITERS = 20


def bench_flashinfer(B, T, h_qk, h_v, d):
    device = "cuda"
    dtype = torch.float16
    q = torch.randn((B, T, h_qk, d), dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(B, T, h_qk, d, dtype=torch.float32, device=device), p=2, dim=-1
    ).to(dtype)
    v = torch.randn((B, T, h_v, d), dtype=dtype, device=device)
    g = F.logsigmoid(torch.rand(1, T * B, h_v, dtype=torch.float32, device=device))
    beta = torch.rand(1, T * B, h_v, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn((B, h_v, d, d), dtype=torch.float32, device=device)
    state_out = torch.zeros_like(h0)

    fn = lambda: fi_gdn(q, k, v, g, beta, None, h0, True, None, False, None, state_out)
    times = bench_gpu_time(fn, enable_cupti=True, dry_run_iters=WARMUP, repeat_iters=ITERS)
    torch.cuda.empty_cache()
    return np.average(times)


def bench_fla(B, T, h_qk, h_v, d):
    device = "cuda"
    dtype = torch.float16
    # FLA doesn't support GVA (h_qk != h_v), expand q/k to h_v
    h = h_v
    q = torch.randn((B, T, h, d), dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(B, T, h, d, dtype=torch.float32, device=device), p=2, dim=-1
    ).to(dtype)
    v = torch.randn((B, T, h_v, d), dtype=dtype, device=device)
    g = F.logsigmoid(torch.rand(1, T * B, h_v, dtype=torch.float32, device=device))
    beta = torch.rand(1, T * B, h_v, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn((B, h_v, d, d), dtype=torch.float32, device=device)

    fn = lambda: fla_gdn(q, k, v, g, beta, None, initial_state=h0, output_final_state=True)
    times = bench_gpu_time(fn, enable_cupti=True, dry_run_iters=WARMUP, repeat_iters=ITERS)
    torch.cuda.empty_cache()
    return np.average(times)


def main():
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Model reference: Qwen3.5-397B-A17B (h_k=16, h_v=64, d=128, GVA ratio=4)")
    print()
    header = f"{'Config':<30s}  {'h_qk':>4s} {'h_v':>4s}  {'FlashInfer':>10s}  {'FLA/Triton':>10s}  {'Speedup':>8s}"
    print(header)
    print("─" * len(header))

    for B, T, h_qk, h_v, d, label in CONFIGS:
        fi_ms = bench_flashinfer(B, T, h_qk, h_v, d)
        fla_ms = bench_fla(B, T, h_qk, h_v, d)
        speedup = fla_ms / fi_ms
        marker = "✓" if speedup > 1.0 else "✗"
        print(
            f"{label:<30s}  {h_qk:>4d} {h_v:>4d}  {fi_ms:>9.3f}ms  {fla_ms:>9.3f}ms  {speedup:>7.2f}x {marker}"
        )



if __name__ == "__main__":
    main()
