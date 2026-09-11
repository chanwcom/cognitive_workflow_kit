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

from cwk.loss.pytorch import seq_loss_util, shc_loss
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


def test_long_utterance_target_is_not_all_blank():
    """A long, padded utterance must not get a degenerate all-blank target.

    `calculate_alpha_beta` floors padded label positions at the FINITE
    LOG_0, so they reach `log_gamma = log_alpha + log_beta` at 2 * LOG_0
    = -1413.79. Normalizing log_gamma over the label axis without
    excluding them was only safe while every real alignment scored above
    that floor. It does not, early in training: a long utterance sums T
    per-frame log-probabilities, so around T >= 800 with a not-yet-
    confident head the best real path drops below -1413.79, padding wins
    the logsumexp, every valid position normalizes to roughly -140, exp()
    flushes it to zero, and -- padding labels being clamp(min=0) = blank
    -- the scatter returns "blank with probability 1" for that utterance.

    Guards the shape of the target, not a tolerance: the failure is total
    (all mass on blank, exactly one active class), so the assertions are
    that real classes keep mass at all.
    """
    torch.manual_seed(0)
    num_classes, max_logit_len = 32, 1200
    real_len, padded_len = 180, 210

    labels = torch.full((2, padded_len), -100, dtype=torch.long)
    labels[0, :real_len] = torch.randint(1, num_classes, (real_len,))
    labels[1, :padded_len] = torch.randint(1, num_classes, (padded_len,))
    target_lens = (labels >= 0).sum(1)

    # A freshly-initialized head: near-uniform over the vocabulary, which
    # is what drives the per-frame log-probabilities low enough to matter.
    logits = torch.randn(2, max_logit_len, num_classes) * 0.01
    log_probs = torch.log_softmax(logits, dim=-1)
    logit_lens = torch.full((2,), max_logit_len, dtype=torch.long)

    log_probs.requires_grad_(True)
    loss = shc_loss.ShcLoss.apply(
        labels, target_lens, log_probs, logit_lens, num_classes,
        0.0, 0.0, False, 0.0, False, "class", "fixed", 1.0, 1.0)
    assert torch.isfinite(loss).all(), loss
    loss.sum().backward()

    # Read the target back out of the production gradient rather than
    # rebuilding it here -- a helper that recomputed the normalization
    # would pass whether or not ShcLoss itself is fixed. ShcLoss forms
    # `gradient = log_probs.exp() - ground_truth_prob`, so:
    target = log_probs.exp().detach() - log_probs.grad
    # Sample 0 is the padded one; sample 1 sets the batch width and so
    # carries no padding, which is what makes it immune.
    mid = max_logit_len // 2
    blank_mass = target[0, mid, 0].item()
    n_active = int((target[0, mid] >= 1e-6).sum())
    assert blank_mass < 0.99, (
        f"target collapsed onto blank: blank mass {blank_mass}")
    assert n_active > 1, f"only {n_active} class carries any target mass"


def _label_space_target_for_fixed_utterance(mate_len):
    """Target for one FIXED utterance, batched against a mate of varying
    length. Only the padded label width changes between calls."""
    gen = torch.Generator().manual_seed(7)
    num_classes, max_logit_len, own_len = 32, 400, 60
    own = torch.randint(1, num_classes, (own_len,), generator=gen)

    if mate_len == 0:
        labels = own.unsqueeze(0)
    else:
        width = max(own_len, mate_len)
        labels = torch.full((2, width), -100, dtype=torch.long)
        labels[0, :own_len] = own
        labels[1, :mate_len] = torch.randint(
            1, num_classes, (mate_len,), generator=gen)
    target_lens = (labels >= 0).sum(1)

    gen2 = torch.Generator().manual_seed(11)
    base = torch.randn(1, max_logit_len, num_classes, generator=gen2) * 0.01
    base[:, :, 0] += 4.0                      # blank-leaning, as in training
    logits = base.expand(labels.shape[0], -1, -1).contiguous()
    log_probs = torch.log_softmax(logits, dim=-1).requires_grad_(True)
    logit_lens = torch.full((labels.shape[0],), max_logit_len,
                            dtype=torch.long)

    loss = shc_loss.ShcLoss.apply(
        labels, target_lens, log_probs, logit_lens, num_classes,
        0.1, 0.0, False, 0.0, False, "label", "fixed", 1.0, 1.0)
    loss.sum().backward()
    return (log_probs.exp().detach() - log_probs.grad)[0]


def test_label_space_target_is_invariant_to_batch_mate_length():
    """An utterance's smoothed target must not depend on its batch-mates.

    In "label" space the smoothed axis is the batch's PADDED
    blank-augmented label width, so the uniform component of SETS used to
    be spread over positions past the utterance's own length. Padded
    labels are clamp(min=0) = blank, so that share scattered onto blank:
    the same utterance drifted by ~4e-2 in target probability purely
    because a longer utterance shared its batch. Fixed by confining the
    uniform to each sample's own label length (`axis_lens`).

    Only bites at beta < 1 -- at beta = 1 the uniform term drops out.
    Class space is unaffected: its axis is the vocabulary, which has no
    padding.
    """
    solo = _label_space_target_for_fixed_utterance(0)
    for mate_len in (60, 120, 240, 400):
        batched = _label_space_target_for_fixed_utterance(mate_len)
        max_diff = (batched - solo).abs().max().item()
        assert max_diff < 1e-6, (
            f"target moved by {max_diff:.3e} when batched against a mate of "
            f"length {mate_len}; it must not depend on batch padding")


def test_empty_transcript_does_not_wreck_the_batch_gradient():
    """An L=0 transcript must not poison the other samples' gradients.

    `to_blank_augmented_labels` returns 2L - 1, so an empty transcript
    reaches the loss with a length of -1. Two things then go wrong, and
    neither is contained by the masks downstream, because they act on the
    gradient by multiplication and both NaN * 0 and 3e10 * 0 survive it:

      - `sequence_mask(-1)` selects no label position, so the whole label
        axis masks to -inf, logsumexp returns -inf, and the normalization
        computes -inf - (-inf) = NaN. The loss scalar still reads finite,
        so a run dies with a loss curve that looks healthy right up to
        the step where every parameter turns to NaN.
      - SETS's `axis_lens` width goes negative, making p_p and u_p
        negative and gamma = beta / p_p explode to about -3e10. Finite,
        but a gradient that size ruins the step just as thoroughly.

    Both are fixed by clamping the length to 1, which leaves position 0
    (blank) valid, so an empty transcript gets the target "all mass on
    blank" -- finite, and the right answer for an empty transcript.

    LibriSpeech and libri-light never ship an empty transcript, so this
    needs a corrupt shard, a tokenizer returning nothing, or a new
    corpus. Low probability, unbounded cost.
    """
    for space in ("class", "label", "hybrid"):
        for beta in (0.0, 0.5, 1.0):
            torch.manual_seed(0)
            logits = (torch.randn(2, 50, 32) * 0.1).requires_grad_(True)
            labels = torch.full((2, 10), -1, dtype=torch.long)
            labels[1, :10] = torch.randint(1, 32, (10,))
            loss = shc_loss.ShcLoss.apply(
                labels, torch.tensor([0, 10]), logits.log_softmax(-1),
                torch.full((2,), 50), 32, 0.06, beta, False, 0.0, False,
                space).mean()
            loss.backward()
            where = f"space={space} beta={beta}"
            assert torch.isfinite(loss), where
            assert torch.isfinite(logits.grad).all(), where
            # Guard the magnitude too: the axis_lens half of this failed
            # finitely, so a finiteness check alone would have missed it.
            assert logits.grad.abs().max() < 10.0, (
                f"{where}: gradient blew up to "
                f"{logits.grad.abs().max().item():.3g}")


if __name__ == "__main__":
    # Allow running without pytest installed.
    tests = [
        test_batched_result_matches_solo_computation,
        test_no_padding_sample_unaffected,
        test_long_utterance_target_is_not_all_blank,
        test_label_space_target_is_invariant_to_batch_mate_length,
        test_empty_transcript_does_not_wreck_the_batch_gradient,
    ]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
