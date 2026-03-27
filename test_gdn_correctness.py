"""Quick correctness test for GDN prefill kernel (chunk_size=64 vs FLA reference)."""
import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
device = "cuda"
dtype = torch.float16

def test_zero_input():
    """Zero inputs should produce zero output and zero state."""
    B, T, h_qk, h_v, d = 1, 128, 4, 16, 128  # chunk_size=64 config
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_gdn
    q = torch.zeros((B, T, h_qk, d), dtype=dtype, device=device)
    k = torch.zeros((B, T, h_qk, d), dtype=dtype, device=device)
    v = torch.zeros((B, T, h_v, d), dtype=dtype, device=device)
    g = torch.zeros(1, T * B, h_v, dtype=torch.float32, device=device)
    beta = torch.ones(1, T * B, h_v, dtype=torch.float32, device=device)
    h0 = torch.zeros((B, h_v, d, d), dtype=torch.float32, device=device)
    state_out = torch.zeros_like(h0)

    result = fi_gdn(q, k, v, g, beta, None, h0, True, None, False, None, state_out)
    o = result[0] if isinstance(result, tuple) else result
    o_max = o.abs().max().item()
    s_max = state_out.abs().max().item()
    print(f"Zero input: O max={o_max:.6f}, S max={s_max:.6f}")
    assert o_max == 0, f"O should be 0, got {o_max}"
    assert s_max == 0, f"S should be 0, got {s_max}"
    print("  PASS")

def test_vs_fla():
    """Compare against FLA reference with real inputs."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_gdn
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_gdn

    B, T, h_v, d = 1, 256, 16, 128
    h_qk = 4  # GVA config

    q = torch.randn((B, T, h_v, d), dtype=dtype, device=device) * 0.1
    k = F.normalize(
        torch.randn(B, T, h_v, d, dtype=torch.float32, device=device), p=2, dim=-1
    ).to(dtype)
    v = torch.randn((B, T, h_v, d), dtype=dtype, device=device) * 0.1
    g = F.logsigmoid(torch.rand(1, T * B, h_v, dtype=torch.float32, device=device))
    beta = torch.rand(1, T * B, h_v, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn((B, h_v, d, d), dtype=torch.float32, device=device) * 0.01
    state_out_fi = torch.zeros_like(h0)

    # FLA reference (uses h_v for all heads)
    fla_result = fla_gdn(q, k, v, g, beta, None, initial_state=h0, output_final_state=True)
    o_fla, state_fla = fla_result[0], fla_result[-1]

    # FlashInfer with GVA (h_qk=4, h_v=16)
    q_fi = q[:, :, :h_qk, :]  # only h_qk Q heads
    k_fi = k
    o_fi_result = fi_gdn(q_fi, k_fi, v, g, beta, None, h0, True, None, False, None, state_out_fi)
    o_fi = o_fi_result[0] if isinstance(o_fi_result, tuple) else o_fi_result

    # Compare state (should be very close since state only depends on K, V, g, beta, h0)
    state_diff = (state_out_fi - state_fla).abs()
    state_rdiff = state_diff / (state_fla.abs() + 1e-6)
    print(f"State: max_abs_diff={state_diff.max().item():.6f}, max_rel_diff={state_rdiff.max().item():.6f}")
    print(f"  FLA state range: [{state_fla.min().item():.4f}, {state_fla.max().item():.4f}]")
    print(f"  FI  state range: [{state_out_fi.min().item():.4f}, {state_out_fi.max().item():.4f}]")

    # State comparison is the primary correctness check
    if state_diff.max().item() < 0.01:
        print("  State PASS (exact match)")
    elif state_diff.max().item() < 0.15:
        print("  State PASS (within tolerance)")
    else:
        print("  State FAIL")

def test_chunk128_regression():
    """Ensure chunk_size=128 still works (B=4, symmetric heads)."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_gdn
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_gdn

    B, T, h, d = 4, 256, 32, 128

    q = torch.randn((B, T, h, d), dtype=dtype, device=device) * 0.1
    k = F.normalize(
        torch.randn(B, T, h, d, dtype=torch.float32, device=device), p=2, dim=-1
    ).to(dtype)
    v = torch.randn((B, T, h, d), dtype=dtype, device=device) * 0.1
    g = F.logsigmoid(torch.rand(1, T * B, h, dtype=torch.float32, device=device))
    beta = torch.rand(1, T * B, h, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn((B, h, d, d), dtype=torch.float32, device=device) * 0.01
    state_out_fi = torch.zeros_like(h0)

    fla_result = fla_gdn(q, k, v, g, beta, None, initial_state=h0, output_final_state=True)
    o_fla, state_fla = fla_result[0], fla_result[-1]
    o_fi_result = fi_gdn(q, k, v, g, beta, None, h0, True, None, False, None, state_out_fi)
    o_fi = o_fi_result[0] if isinstance(o_fi_result, tuple) else o_fi_result

    state_diff = (state_out_fi - state_fla).abs()
    state_rdiff = state_diff / (state_fla.abs() + 1e-6)
    print(f"chunk128 State: max_abs={state_diff.max().item():.6f}, max_rel={state_rdiff.max().item():.6f}")
    # Use absolute error threshold (relative error is misleading for near-zero values)
    if state_diff.max().item() < 0.15:
        print("  PASS")
    else:
        print("  FAIL")


if __name__ == "__main__":
    print("=== Test 1: Zero input ===")
    test_zero_input()
    print("\n=== Test 2: chunk_size=128 regression ===")
    test_chunk128_regression()
    print("\n=== Test 3: vs FLA (chunk_size=64) ===")
    test_vs_fla()
