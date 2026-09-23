"""Tests for apply_peak_gated_fas_smoothing.

Run as `python shc_loss_peak_gate_test.py` -- pytest is not installed in
py3_12_sets, and the rest of this directory mixes both styles.
"""
import torch

import shc_loss_util as u

ALPHA, BETA, EPS = 0.10, 0.0, 1e-10


def _fixture():
    """(B=1, T=4, K=5) with one frame per peak regime.

    frame 0: peak 0.97  -- above any sane threshold
    frame 1: peak 0.90  -- exactly at 0.9, so it pins down >= vs >
    frame 2: peak 0.50  -- clearly below
    frame 3: padded
    """
    y = torch.tensor([[[0.97, 0.02, 0.01, 0.0, 0.0],
                       [0.90, 0.06, 0.04, 0.0, 0.0],
                       [0.50, 0.30, 0.20, 0.0, 0.0],
                       [0.25, 0.25, 0.25, 0.25, 0.0]]], dtype=torch.float64)
    return y, torch.tensor([3])


def test_gates_partition_the_frames():
    """high + low must reconstruct plain FAS exactly, frame by frame.

    If the two gates overlapped, some frame would be smoothed in both
    runs; if they left a hole, some frame would be smoothed in neither.
    Either way a matched pair of runs would stop being a decomposition of
    FAS and the experiment could not be read as "which frames carry the
    effect".
    """
    y, lens = _fixture()
    plain = u.apply_floored_active_support_smoothing(y, lens, ALPHA, BETA,
                                                     eps=EPS)
    hi = u.apply_peak_gated_fas_smoothing(y, lens, ALPHA, BETA, eps=EPS,
                                          peak_thresh=0.9, gate="high")
    lo = u.apply_peak_gated_fas_smoothing(y, lens, ALPHA, BETA, eps=EPS,
                                          peak_thresh=0.9, gate="low")
    peak = y.max(-1).values
    sel_hi = (peak >= 0.9).unsqueeze(-1)
    merged = torch.where(sel_hi, hi, lo)
    assert torch.allclose(merged, plain, atol=0, rtol=0), (
        f"merged != plain\n{merged}\n{plain}")
    print("PASSED: test_gates_partition_the_frames")


def test_ungated_frames_are_untouched_bitwise():
    """Ungated frames keep the target exactly, not a reduced rate."""
    y, lens = _fixture()
    hi = u.apply_peak_gated_fas_smoothing(y, lens, ALPHA, BETA, eps=EPS,
                                          peak_thresh=0.9, gate="high")
    # frame 2 (peak 0.50) is below the cutoff, so gate="high" skips it.
    assert torch.equal(hi[0, 2], y[0, 2]), f"{hi[0, 2]} != {y[0, 2]}"
    # ... and frame 0 is not.
    assert not torch.equal(hi[0, 0], y[0, 0])
    print("PASSED: test_ungated_frames_are_untouched_bitwise")


def test_threshold_is_inclusive_on_the_high_side():
    """peak == thresh belongs to "high", so the pair stays a partition."""
    y, lens = _fixture()
    hi = u.apply_peak_gated_fas_smoothing(y, lens, ALPHA, BETA, eps=EPS,
                                          peak_thresh=0.9, gate="high")
    lo = u.apply_peak_gated_fas_smoothing(y, lens, ALPHA, BETA, eps=EPS,
                                          peak_thresh=0.9, gate="low")
    assert not torch.equal(hi[0, 1], y[0, 1]), "frame at exactly 0.9 skipped"
    assert torch.equal(lo[0, 1], y[0, 1]), "frame at exactly 0.9 double-counted"
    print("PASSED: test_threshold_is_inclusive_on_the_high_side")


def test_degenerate_thresholds_reduce_to_fas_and_to_nothing():
    """thresh 0 with gate=high is plain FAS; thresh 0 with gate=low is a no-op.

    A run whose gated fraction has drifted to one of these ends has
    stopped testing anything, so the endpoints need to be exact.
    """
    y, lens = _fixture()
    plain = u.apply_floored_active_support_smoothing(y, lens, ALPHA, BETA,
                                                     eps=EPS)
    all_hi = u.apply_peak_gated_fas_smoothing(y, lens, ALPHA, BETA, eps=EPS,
                                              peak_thresh=0.0, gate="high")
    none_lo = u.apply_peak_gated_fas_smoothing(y, lens, ALPHA, BETA, eps=EPS,
                                               peak_thresh=0.0, gate="low")
    valid = (torch.arange(4).unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(-1)
    assert torch.allclose(all_hi, plain, atol=0, rtol=0)
    assert torch.allclose(none_lo, y * valid, atol=0, rtol=0)
    print("PASSED: test_degenerate_thresholds_reduce_to_fas_and_to_nothing")


def test_padded_frames_are_zeroed():
    y, lens = _fixture()
    for gate in ("high", "low"):
        out = u.apply_peak_gated_fas_smoothing(y, lens, ALPHA, BETA, eps=EPS,
                                               peak_thresh=0.9, gate=gate)
        assert torch.all(out[0, 3] == 0.0), f"{gate}: {out[0, 3]}"
    print("PASSED: test_padded_frames_are_zeroed")


def test_bad_gate_raises():
    y, lens = _fixture()
    try:
        u.apply_peak_gated_fas_smoothing(y, lens, ALPHA, BETA, gate="HIGH")
    except ValueError:
        print("PASSED: test_bad_gate_raises")
        return
    raise AssertionError("expected ValueError for gate='HIGH'")


if __name__ == "__main__":
    test_gates_partition_the_frames()
    test_ungated_frames_are_untouched_bitwise()
    test_threshold_is_inclusive_on_the_high_side()
    test_degenerate_thresholds_reduce_to_fas_and_to_nothing()
    test_padded_frames_are_zeroed()
    test_bad_gate_raises()
    print("\nAll peak-gate tests passed.")
