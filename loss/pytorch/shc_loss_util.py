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

def apply_post_processing(est_probs, logits_len, alpha, beta, eps):
    """Applies masked-uniform label smoothing to a batch of probs.

    Vectorized "masked-uniform" label smoothing post-processing.

    Implements:
        y_ls(b, t) = (1 - alpha) * y(b, t)
                     + alpha * (beta * u + gamma * u_p(b, t))
        gamma       = (1 - beta) / p_p(b, t)

    The (beta * u + gamma * u_p) term is itself a valid probability
    distribution (sums to 1 over classes): sum(u) = 1 and
    sum(u_p) = p_p, so gamma must divide by p_p (not multiply) for
    beta + gamma * p_p to equal 1.

    where, for each (b, t):
        - y(b, t)   : the C-dim probability vector est_probs[b, t, :].
        - p_p(b, t) : fraction of classes with prob >= eps
                      (N_p / C).
        - u_p(b, t) : masked uniform dist. 1/C on classes with
                      prob >= eps, 0 elsewhere.
        - u         : plain uniform dist, 1/C everywhere.

    Positions t >= logits_len[b] are padding and are zeroed out in
    the output (don't-care region).
    Args:
        est_probs: Float tensor of shape (B, T, C). Probability
            distributions over C classes, per batch/time step.
        logits_len: Long tensor of shape (B,). Valid (unpadded)
            length of each sequence in the batch.
        alpha: Python float in [0, 1]. Overall smoothing weight
            mixed in with (beta * u + gamma * u_p). Fixed scalar.
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

    # gamma(b, t) = (1 - beta) / p_p(b, t), shape (B, T, 1).
    # Division (not multiplication) is required so that
    # mix = beta * u + gamma * u_p sums to 1 over the class axis:
    # sum(u) = 1 and sum(u_p) = p_p, so sum(mix) =
    # beta + gamma * p_p, which only equals 1 when
    # gamma = (1 - beta) / p_p. p_p is guaranteed to be > 0
    # since every valid (b, t) row sums to 1, so at least one
    # class must be >= eps; a tiny clamp is kept only as a
    # numerical safety net.
    gamma = (1.0 - beta) / p_p.clamp(min=1e-12)

    # Mix the plain uniform and masked-uniform components, then
    # blend with the original distribution y(b, t).
    smoothed = beta * u + gamma * u_p
    y_ls = (1.0 - alpha) * est_probs + alpha * smoothed

    # Build a (B, T) validity mask from logits_len and zero out
    # any padded time steps (t >= logits_len[b]).
    time_idx = torch.arange(t, device=est_probs.device)
    valid = time_idx.unsqueeze(0) < logits_len.unsqueeze(1)  # (B, T)
    y_ls = y_ls * valid.unsqueeze(-1).to(est_probs.dtype)

    return y_ls
