"""A module implementing utilities for sequence losses."""

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
        est_probs: Float tensor of shape (B, T, C). Probability
            distributions over C classes, per batch/time step.
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
        Float tensor of shape (B, T, C), same shape/dtype as
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
        est_probs: Float tensor of shape (B, T, C). Probability
            distributions over C classes, per batch/time step.
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
        Float tensor of shape (B, T, C), same shape/dtype as
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
        - y(b, t)   : the C-dim probability vector est_probs[b, t, :].
        - p_p(b, t) : fraction of classes with prob >= eps
                      (N_p / C).
        - u_p(b, t) : masked uniform dist. 1/C on classes with
                      prob >= eps, 0 elsewhere.
        - u         : plain uniform dist, 1/C everywhere.

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

    # u_p(b, t, c) = 1/C where prob >= eps, else 0.
    u_p = ge_mask.to(est_probs.dtype) / c

    # u is the plain uniform distribution, a scalar broadcastable
    # constant (1/C on every class).
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
    ("peak") class of each (b, t) distribution is left completely
    untouched; only the remaining probability mass is smoothed.

    Args:
        est_probs: Float tensor of shape (B, T, C). Probability
            distributions over C classes, per batch/time step.
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
        Float tensor of shape (B, T, C), same shape/dtype as
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

    Args:
        est_probs: Float tensor of shape (B, T, C). Probability
            distributions over C classes, per batch/time step.
        logits_len: Long tensor of shape (B,). Valid (unpadded)
            length of each sequence in the batch.
        alpha: Python float in [0, 1]. Defines the confidence cap
            (1 - alpha) on the peak class. Fixed scalar.
        eps: Small positive float threshold (e.g. 1e-10) used to
            decide whether a non-peak class counts as "active"
            (prob >= eps).

    Returns:
        Float tensor of shape (B, T, C), same shape/dtype as
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
