"""A module implementing utilities for sequence losses.

===============================================================================
WHICH AXIS IS BEING SMOOTHED? (read this before editing anything below)
===============================================================================

Every smoothing function here operates generically on the LAST axis of its
input. It does NOT know or care what that axis means. The docstrings below
therefore call it "K categories".

`ShcLoss.forward` can drive these functions in either of two spaces, selected
by its `smoothing_space` argument:

  * "label" (the historical default, and what every SETS / PP-SETS / PC-SETS
    experiment run so far used): the input is `gamma` with shape (B, T, L),
    where L is the BLANK-AUGMENTED LABEL SEQUENCE length. The last axis
    indexes label *positions*, not classes. Smoothing happens BEFORE the
    scatter_add_ that maps L -> C.

  * "class": the input is `ground_truth_prob` with shape (B, T, C), where C
    is the vocabulary size. Smoothing happens AFTER that scatter_add_, so
    the last axis indexes actual output classes.

Both are mathematically valid -- the smoothed row sums to 1 either way, and
the resulting cross-entropy target is a proper distribution over classes in
both cases. But they are genuinely DIFFERENT algorithms, and the difference
is large. It matters enough that it must be described accurately in any
write-up, so it is spelled out here.

Concretely, ShcLoss builds the augmented label sequence as
[t0, blank, t1, blank, ..., t_{U-1}] (length L = 2U - 1, see
`seq_loss_util.to_blank_augmented_labels` with boundary_blanks=False), so
roughly HALF of all label positions are blank. A distribution that is
uniform over the L axis is therefore very far from uniform over classes.
Measured for a 20-token transcript with C = 32 (L = 39):

    "plain uniform" (beta = 0), i.e. 1/L on every label position
      -> in class space: blank = 0.487, only the 15 classes that occur in
         the transcript get any mass at all (each proportional to how many
         times it occurs), and the other 17 classes get EXACTLY 0.
      -> a true uniform over classes would instead be 0.031 on all 32.
      -> entropy 2.00 nats, vs. log(32) = 3.47 for a true class uniform.

    "masked uniform" (beta = 1), i.e. 1/N_p on the label positions that are
    reachable at frame t
      -> in class space: blank = 0.47..0.60 depending on how wide the
         reachable band is, with only a handful of non-blank classes
         receiving mass and the remaining ~25-29 classes getting 0.

So in the "label" space, what actually gets mixed into the target is not a
uniform prior but a blank-heavy, occurrence-count-weighted prior supported
only on classes present in that utterance's transcript. That is much closer
to unigram label smoothing (with a strong CTC blank prior) than to classical
uniform label smoothing -- arguably a better prior for CTC, but definitely
not the same thing, and NOT what the phrase "smooth toward the uniform
distribution" would lead a reader to expect.

Note also that `peak = argmax(dim=-1)` means different things in the two
spaces: in "label" space it is the most probable label POSITION, which need
not correspond to the most probable CLASS (blank's mass is split across the
~L/2 blank positions, so blank can dominate in class space while losing the
per-position argmax). `apply_peak_preserving_...` and
`apply_peak_capping_...` inherit this, so PC-SETS's `y[peak] > 1 - alpha`
gate triggers on a systematically smaller quantity in "label" space than a
class-space reading of the same threshold would suggest.
===============================================================================
"""

# pylint: disable=no-member, invalid-name, import-error


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"
# Standard imports

# Third-party imports
import torch

# TODO(chanwcom) Replace with this one. But unit tests need to be updated.
#LOG_00 = torch.tensor(np.log(np.finfo(np.float64).tiny).astype(np.float32))

LOG_0 = -706.893623  # float(np.log(1e-307))

def apply_post_processing(est_probs, logits_len, alpha, beta, eps=1e-6,
                           peak_preserving=False, gamma=None,
                           peak_capping=False):
    """Dispatches to the selected post-processing variant.

    Args:
        est_probs: Float tensor of shape (B, T, K). Probability
            distributions over the last axis's K categories, per
            batch/time step. NOTE: K is NOT necessarily the vocabulary
            size -- see the module docstring. ShcLoss passes the
            blank-augmented label axis (K = L) in "label" space and the
            class axis (K = C) in "class" space.
        logits_len: Long tensor of shape (B,). Valid (unpadded)
            length of each sequence in the batch.
        alpha: Python float in [0, 1]. Overall smoothing weight when
            neither peak_preserving nor peak_capping is set (used by
            SETS); confidence-cap parameter (cap = 1 - alpha) when
            peak_capping is set. Fixed scalar.
        beta: Python float in [0, 1]. Fixed scalar. Used only by
            SETS (peak_preserving and peak_capping both False).
        eps: Small positive float threshold (e.g. 1e-10) used to
            decide whether a class probability counts as
            "active" (prob >= eps).
        peak_preserving: If True, uses
            apply_peak_preserving_selective_estimated_target_smoothing
            (driven by `gamma`, alpha/beta ignored).
        gamma: Python float in [0, 1]. Fraction of each non-peak
            class's probability mass redistributed uniformly over
            the other active, non-peak classes. Fixed scalar. Used
            only when peak_preserving is True.
        peak_capping: If True (and peak_preserving is False), uses
            apply_peak_capping_selective_estimated_target_smoothing
            (driven by `alpha` as the cap parameter; beta/gamma
            ignored). Ignored if peak_preserving is True.

    Returns:
        Float tensor of shape (B, T, K), same shape/dtype as
        est_probs, with smoothing applied. Time steps beyond
        logits_len are set to 0.
    """
    if peak_preserving:
        return apply_peak_preserving_selective_estimated_target_smoothing(
            est_probs, logits_len, gamma, eps)
    if peak_capping:
        return apply_peak_capping_selective_estimated_target_smoothing(
            est_probs, logits_len, alpha, eps)
    return apply_selective_estimated_target_smoothing(
        est_probs, logits_len, alpha, beta, eps)


def apply_selective_estimated_target_smoothing(est_probs, logits_len, alpha,
                                                beta, eps=1e-6):
    """Applies masked-uniform label smoothing to a batch of probs.

    Args:
        est_probs: Float tensor of shape (B, T, K). Probability
            distributions over the last axis's K categories, per
            batch/time step. NOTE: K is NOT necessarily the vocabulary
            size -- see the module docstring. ShcLoss passes the
            blank-augmented label axis (K = L) in "label" space and the
            class axis (K = C) in "class" space.
        logits_len: Long tensor of shape (B,). Valid (unpadded)
            length of each sequence in the batch.
        alpha: Python float in [0, 1]. Overall smoothing weight
            mixed in with ((1.0 - beta) * u + gamma * u_p). Fixed scalar.
        beta: Python float in [0, 1]. Weight of the plain uniform
            component vs. the masked-uniform component. Fixed
            scalar.
        eps: Small positive float threshold (e.g. 1e-10) used to
            decide whether a class probability counts as
            "active" (prob >= eps) for both p_p and u_p.

    Returns:
        Float tensor of shape (B, T, K), same shape/dtype as
        est_probs, with smoothing applied. Time steps beyond
        logits_len are set to 0.

    Vectorized "Selective Estimated Target Smoothing" (SETS)
    post-processing.

    Implements:
        y_ls(b, t) = (1 - alpha) * y(b, t)
                     + alpha * ((1.0 - beta) * u + gamma * u_p(b, t))
        gamma       = beta / p_p(b, t)

    The ((1.0 - beta) * u + gamma * u_p) term is itself a valid probability
    distribution (sums to 1 over classes): sum(u) = 1 and
    sum(u_p) = p_p, so gamma must divide by p_p (not multiply) for
    (1.0 - beta) + gamma * p_p to equal 1.

    where, for each (b, t):
        - y(b, t)   : the K-dim probability vector est_probs[b, t, :].
        - p_p(b, t) : fraction of categories with prob >= eps
                      (N_p / K).
        - u_p(b, t) : masked uniform dist. 1/K on categories with
                      prob >= eps, 0 elsewhere.
        - u         : uniform dist over the last axis, 1/K everywhere.

    IMPORTANT: "uniform" here means uniform over the LAST AXIS, which is
    only the class axis in "class" space. In the historical "label"
    space, u is uniform over blank-augmented label POSITIONS, which in
    class space is a blank-heavy, occurrence-count-weighted prior
    supported only on the classes present in the transcript -- roughly
    blank = 0.49 rather than 1/32 = 0.031. See the module docstring for
    measured numbers.

    Positions t >= logits_len[b] are padding and are zeroed out in
    the output (don't-care region).
    """
    b, t, c = est_probs.shape

    # Single boolean mask (non-strict ">=") shared by both p_p and
    # u_p, as they now use the same activity threshold.
    ge_mask = est_probs >= eps  # (B, T, C).

    # p_p(b, t) = N_p / C, kept as (B, T, 1) for broadcasting.
    n_p = ge_mask.sum(dim=-1, keepdim=True).to(est_probs.dtype)
    p_p = n_p / c

    # u_p(b, t, k) = 1/K where prob >= eps, else 0.
    u_p = ge_mask.to(est_probs.dtype) / c

    # u is uniform over the LAST AXIS, a scalar broadcastable constant
    # (1/K on every category of that axis -- which is a label position,
    # not a class, in "label" space; see the module docstring).
    u = 1.0 / c

    # gamma(b, t) = beta / p_p(b, t), shape (B, T, 1).
    # Division (not multiplication) is required so that
    # mix = (1.0 - beta) * u + gamma * u_p sums to 1 over the class axis:
    # sum(u) = 1 and sum(u_p) = p_p, so sum(mix) =
    # (1.0 - beta) + gamma * p_p, which only equals 1 when
    # gamma =beta / p_p. p_p is guaranteed to be > 0
    # since every valid (b, t) row sums to 1, so at least one
    # class must be >= eps; a tiny clamp is kept only as a
    # numerical safety net.
    gamma = beta / p_p.clamp(min=1e-12)

    # Mix the plain uniform and masked-uniform components, then
    # blend with the original distribution y(b, t).
    smoothed = (1.0 - beta) * u + gamma * u_p
    y_ls = (1.0 - alpha) * est_probs + alpha * smoothed

    # Build a (B, T) validity mask from logits_len and zero out
    # any padded time steps (t >= logits_len[b]).
    time_idx = torch.arange(t, device=est_probs.device)
    valid = time_idx.unsqueeze(0) < logits_len.unsqueeze(1)  # (B, T)
    y_ls = y_ls * valid.unsqueeze(-1).to(est_probs.dtype)

    return y_ls


def apply_peak_preserving_selective_estimated_target_smoothing(
        est_probs, logits_len, gamma, eps=1e-6):
    """Peak-preserving variant of Selective Estimated Target Smoothing.

    Unlike apply_selective_estimated_target_smoothing, the argmax
    ("peak") category of each (b, t) distribution is left completely
    untouched; only the remaining probability mass is smoothed.

    NOTE: "peak" is the argmax of the LAST AXIS. In "label" space that
    is the most probable label POSITION, which is not necessarily the
    most probable CLASS -- blank's mass is spread over the ~L/2 blank
    positions, so blank can be the dominant class while still losing
    the per-position argmax. See the module docstring.

    Args:
        est_probs: Float tensor of shape (B, T, K). Probability
            distributions over the last axis's K categories, per
            batch/time step. NOTE: K is NOT necessarily the vocabulary
            size -- see the module docstring. ShcLoss passes the
            blank-augmented label axis (K = L) in "label" space and the
            class axis (K = C) in "class" space.
        logits_len: Long tensor of shape (B,). Valid (unpadded)
            length of each sequence in the batch.
        gamma: Python float in [0, 1]. Fraction of each non-peak
            class's probability mass to redistribute uniformly
            over the other active (prob >= eps), non-peak classes.
            Fixed scalar.
        eps: Small positive float threshold (e.g. 1e-10) used to
            decide whether a non-peak class counts as "active"
            (prob >= eps).

    Returns:
        Float tensor of shape (B, T, K), same shape/dtype as
        est_probs, with smoothing applied. Time steps beyond
        logits_len are set to 0.

    Implements, for each (b, t), with peak = argmax(y),
    S_rest = 1 - y[peak], A' = {c != peak : y[c] >= eps},
    N' = |A'|:
        y_new[peak]                    = y[peak]
        y_new[c], c in A'              = (1-gamma)*y[c]
                                          + gamma * S_rest / N'
        y_new[c], c not in A', c!=peak = (1-gamma)*y[c]

    sum(y_new) = y[peak] + (1-gamma)*S_rest + gamma*S_rest
               = y[peak] + S_rest = 1, independent of gamma.

    If N' == 0 (the peak is the only active class), there is
    nothing to redistribute onto, so y_new = y unchanged for that
    (b, t).
    """
    b, t, c = est_probs.shape

    # One-hot mask of the argmax ("peak") class per (b, t).
    peak_idx = est_probs.argmax(dim=-1)  # (B, T).
    peak_mask = torch.nn.functional.one_hot(
        peak_idx, num_classes=c).to(dtype=torch.bool)  # (B, T, C).

    peak_prob = torch.gather(
        est_probs, -1, peak_idx.unsqueeze(-1))  # (B, T, 1).
    s_rest = 1.0 - peak_prob  # (B, T, 1).

    # Non-peak classes with prob >= eps: the redistribution target
    # set A'.
    active_non_peak_mask = (est_probs >= eps) & ~peak_mask  # (B, T, C).
    n_prime = active_non_peak_mask.sum(
        dim=-1, keepdim=True).to(est_probs.dtype)  # (B, T, 1).

    # Guard against N' == 0; the resulting per-class term is
    # discarded for those (b, t) anyway (see the torch.where below),
    # so clamping just avoids a 0/0 NaN.
    per_class_share = gamma * s_rest / n_prime.clamp(min=1.0)  # (B, T, 1).

    scaled = (1.0 - gamma) * est_probs
    y_new = scaled + active_non_peak_mask.to(est_probs.dtype) * per_class_share
    # Restore the peak class exactly (undo the (1-gamma) scaling above).
    y_new = torch.where(peak_mask, est_probs, y_new)
    # N' == 0: nothing to redistribute onto -> leave y unchanged.
    y_new = torch.where(n_prime == 0, est_probs, y_new)

    # Build a (B, T) validity mask from logits_len and zero out
    # any padded time steps (t >= logits_len[b]).
    time_idx = torch.arange(t, device=est_probs.device)
    valid = time_idx.unsqueeze(0) < logits_len.unsqueeze(1)  # (B, T)
    y_new = y_new * valid.unsqueeze(-1).to(est_probs.dtype)

    return y_new


def apply_peak_capping_selective_estimated_target_smoothing(
        est_probs, logits_len, alpha, eps=1e-6):
    """Peak-capping variant of Selective Estimated Target Smoothing.

    Unlike apply_peak_preserving_selective_estimated_target_smoothing
    (which never touches the peak), this variant only intervenes when
    the peak's probability exceeds a confidence cap of (1 - alpha);
    (b, t) positions where the peak is already at or below that cap
    are left completely untouched.

    NOTE: as in the peak-preserving variant, "peak" is the argmax of
    the LAST AXIS. This matters more here than there, because the cap
    is a threshold on that value: in "label" space the peak is a single
    label POSITION, whose probability is systematically smaller than
    the corresponding class probability (blank's mass is split across
    the ~L/2 blank positions). The same numeric cap therefore fires
    considerably less often in "label" space than a class-space reading
    of `y[peak] > 1 - alpha` would suggest. See the module docstring.

    Args:
        est_probs: Float tensor of shape (B, T, K). Probability
            distributions over the last axis's K categories, per
            batch/time step. NOTE: K is NOT necessarily the vocabulary
            size -- see the module docstring. ShcLoss passes the
            blank-augmented label axis (K = L) in "label" space and the
            class axis (K = C) in "class" space.
        logits_len: Long tensor of shape (B,). Valid (unpadded)
            length of each sequence in the batch.
        alpha: Python float in [0, 1]. Defines the confidence cap
            (1 - alpha) on the peak class. Fixed scalar.
        eps: Small positive float threshold (e.g. 1e-10) used to
            decide whether a non-peak class counts as "active"
            (prob >= eps).

    Returns:
        Float tensor of shape (B, T, K), same shape/dtype as
        est_probs, with capping applied. Time steps beyond
        logits_len are set to 0.

    Implements, for each (b, t), with peak = argmax(y):
        if y[peak] > (1 - alpha):
            k = (1 - alpha) / y[peak]
            scaled = k * y                    (whole row scaled by k)
            lost_mass = 1 - k
            A' = {c != peak : y[c] >= eps}, N' = |A'|
            y_new[peak]                    = scaled[peak]  (= 1-alpha)
            y_new[c], c in A'              = scaled[c] + lost_mass / N'
            y_new[c], c not in A', c!=peak = scaled[c]
        else:
            y_new = y

    sum(y_new) = k * sum(y) + lost_mass = k + (1 - k) = 1 when the cap
    is applied (sum(y) = 1 by assumption), and trivially 1 otherwise.

    If N' == 0 (the peak is the only active class), there is nothing
    to redistribute lost_mass onto, so y_new = y unchanged for that
    (b, t) (same fallback as the peak-preserving variant).
    """
    b, t, c = est_probs.shape

    peak_idx = est_probs.argmax(dim=-1)  # (B, T).
    peak_mask = torch.nn.functional.one_hot(
        peak_idx, num_classes=c).to(dtype=torch.bool)  # (B, T, C).
    peak_prob = torch.gather(
        est_probs, -1, peak_idx.unsqueeze(-1))  # (B, T, 1).

    exceeds = peak_prob > (1.0 - alpha)  # (B, T, 1) bool.

    # k(b, t) = (1-alpha) / peak_prob(b, t); only meaningful where
    # `exceeds` is True (peak_prob is guaranteed > 0 there since it's
    # the max of a valid probability row, but clamp defensively).
    k = (1.0 - alpha) / peak_prob.clamp(min=1e-12)  # (B, T, 1).
    scaled = k * est_probs
    lost_mass = 1.0 - k  # (B, T, 1).

    active_non_peak_mask = (est_probs >= eps) & ~peak_mask  # (B, T, C).
    n_prime = active_non_peak_mask.sum(
        dim=-1, keepdim=True).to(est_probs.dtype)  # (B, T, 1).
    per_class_share = lost_mass / n_prime.clamp(min=1.0)  # (B, T, 1).

    capped = (scaled +
             active_non_peak_mask.to(est_probs.dtype) * per_class_share)
    # N' == 0: nothing to redistribute lost_mass onto -> leave y
    # unchanged (matches the peak-preserving variant's fallback).
    capped = torch.where(n_prime == 0, est_probs, capped)
    # Cap not exceeded -> leave y unchanged.
    y_new = torch.where(exceeds, capped, est_probs)

    # Build a (B, T) validity mask from logits_len and zero out
    # any padded time steps (t >= logits_len[b]).
    time_idx = torch.arange(t, device=est_probs.device)
    valid = time_idx.unsqueeze(0) < logits_len.unsqueeze(1)  # (B, T)
    y_new = y_new * valid.unsqueeze(-1).to(est_probs.dtype)

    return y_new
