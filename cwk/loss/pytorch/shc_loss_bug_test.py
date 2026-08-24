"""Regression test for the log_beta padding-contamination fix.

Bug summary
-----------
Inside `calculate_alpha_beta`'s backward recursion:

    log_beta[:, t_b, :] = logsumexp(
        (log_beta[:, t_b + 1, :] + log_target_probs[:, t_b + 1, :])
        .unsqueeze(1) + trans_mask, dim=2)

For a sample `b` whose real length is shorter than the batch's
`max_logit_len`, the step where `t_b == logit_lens[b] - 1` (i.e. its true
last valid frame) reads `log_target_probs[:, t_b + 1, :]`, which is a
*padded* time frame. That padded value is not masked out anywhere before
this point, so it "leaks" into what should be a clean boundary condition.

This is fixed unconditionally in `calculate_alpha_beta` via the
`current_mask` blend: when `t_b + 1` falls outside a sample's valid
`logit_lens`, `log_beta[:, t_b, :]` is pinned to `initial_log_beta` (the
canonical boundary condition) instead of using the contaminated value.
There is no longer a toggle to select the old, buggy behavior -- the fix
is always applied -- so this test only checks the (single) current
implementation.

IMPORTANT: `log_seq_prob_final` is read only from `log_alpha`
(`log_alpha[batch, logit_lens-1, target_lens-1]`), and the forward alpha
recursion never looks at *future* frames -- so `log_seq_prob_final` (and
therefore the scalar `loss = -log_seq_prob_final`) was never affected by
this bug. The corruption was isolated to `log_beta`, which feeds into
`log_gamma = log_alpha + log_beta` and from there into the *gradient*
(`ShcLoss.backward`), not the loss value itself. So the test below checks
`log_beta` directly (specifically at each padded sample's true last valid
frame, `t = logit_lens[b] - 1`), not `log_seq_prob_final`.

Invariance under test
----------------------
Computing `log_beta` at a sequence's true last valid frame should not depend
on whether it happens to share a batch with a longer (padded) sequence: the
batched-with-padding result must match the solo (no padding) result for the
shorter sequence.
"""
import torch

from cwk.loss.pytorch.shc_loss import calculate_alpha_beta, LOG_0


def _make_step_trans_mask(batch_size: int, max_target_len: int) -> torch.Tensor:
    """Simple CTC-style transition mask: self-loop (i->i) and step (i->i+1)."""
    trans = torch.full((max_target_len, max_target_len), LOG_0)
    idx = torch.arange(max_target_len)
    trans[idx, idx] = 0.0
    if max_target_len > 1:
        trans[idx[:-1], idx[1:]] = 0.0
    return trans.unsqueeze(0).expand(batch_size, -1, -1).clone()


def _random_log_target_probs(batch_size, max_logit_len, max_target_len,
                              seed):
    """Produces a valid (batch, T, L) log-probability-like tensor via
    log_softmax over a random logits tensor, so values are realistic."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(
        batch_size, max_logit_len, max_target_len, generator=g
    )
    return torch.log_softmax(logits, dim=-1)


def test_batched_result_matches_solo_computation():
    """Batched (padded) result == solo (unpadded) result at the true last
    valid frame -- this is exactly the frame the padding-contamination bug
    used to corrupt."""
    torch.manual_seed(0)

    max_target_len = 3   # both samples use the same target length here
    max_logit_len = 5    # sample A uses the full length
    short_logit_len = 3  # sample B is shorter -> gets padded

    logit_lens = torch.tensor([max_logit_len, short_logit_len])
    target_lens = torch.tensor([max_target_len, max_target_len])

    trans_mask = _make_step_trans_mask(2, max_target_len)
    log_target_probs = _random_log_target_probs(
        2, max_logit_len, max_target_len, seed=42
    )

    # --- Batched run (sample B is padded out to max_logit_len=5) ---
    _, log_beta_batched, _ = calculate_alpha_beta(
        trans_mask, log_target_probs, target_lens, logit_lens,
    )

    # --- Solo run: sample B alone, with NO padding at all (T == 3) ---
    solo_trans_mask = trans_mask[1:2]
    solo_log_target_probs = log_target_probs[1:2, :short_logit_len, :]
    solo_logit_lens = torch.tensor([short_logit_len])
    solo_target_lens = target_lens[1:2]

    _, log_beta_solo, _ = calculate_alpha_beta(
        solo_trans_mask, solo_log_target_probs, solo_target_lens,
        solo_logit_lens,
    )

    # Compare log_beta at sample B's TRUE last valid frame
    # (t = short_logit_len - 1 = 2) -- this is exactly the frame the bug
    # corrupts. This is a vector over the target dimension.
    t_last = short_logit_len - 1
    beta_batched_last_frame = log_beta_batched[1, t_last, :]
    beta_solo_last_frame = log_beta_solo[0, t_last, :]

    assert torch.allclose(
        beta_batched_last_frame, beta_solo_last_frame, atol=1e-4
    ), (
        "log_beta should be padding-invariant at the true last valid "
        f"frame, but got batched={beta_batched_last_frame} vs "
        f"solo={beta_solo_last_frame}"
    )


def test_no_padding_sample_unaffected():
    """For the sample that has no padding at all (logit_lens[b] ==
    max_logit_len), computing it solo vs. batched-with-a-shorter-sample
    must give identical results (padding elsewhere must not leak in)."""
    torch.manual_seed(0)

    max_target_len = 3
    max_logit_len = 5
    short_logit_len = 3

    logit_lens = torch.tensor([max_logit_len, short_logit_len])
    target_lens = torch.tensor([max_target_len, max_target_len])

    trans_mask = _make_step_trans_mask(2, max_target_len)
    log_target_probs = _random_log_target_probs(
        2, max_logit_len, max_target_len, seed=42
    )

    log_alpha_batched, log_beta_batched, log_seq_prob_batched = (
        calculate_alpha_beta(
            trans_mask, log_target_probs, target_lens, logit_lens,
        )
    )

    solo_trans_mask = trans_mask[0:1]
    solo_log_target_probs = log_target_probs[0:1]
    solo_logit_lens = logit_lens[0:1]
    solo_target_lens = target_lens[0:1]

    log_alpha_solo, log_beta_solo, log_seq_prob_solo = calculate_alpha_beta(
        solo_trans_mask, solo_log_target_probs, solo_target_lens,
        solo_logit_lens,
    )

    assert torch.allclose(log_seq_prob_batched[0], log_seq_prob_solo[0])
    assert torch.allclose(log_alpha_batched[0], log_alpha_solo[0])
    assert torch.allclose(log_beta_batched[0], log_beta_solo[0])


if __name__ == "__main__":
    # Allow running without pytest installed.
    tests = [
        test_batched_result_matches_solo_computation,
        test_no_padding_sample_unaffected,
    ]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
