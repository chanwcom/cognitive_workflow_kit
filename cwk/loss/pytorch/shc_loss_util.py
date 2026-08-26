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

def _masked_frame_mean(per_frame, logits_len):
    """Averages a (B, T) per-frame quantity over each sequence's valid frames.

    Args:
        per_frame: Float tensor of shape (B, T).
        logits_len: Long tensor of shape (B,).

    Returns:
        Float tensor of shape (B,).
    """
    max_time = per_frame.shape[1]
    time_idx = torch.arange(max_time, device=per_frame.device)
    valid = (time_idx.unsqueeze(0) < logits_len.unsqueeze(1)).to(
        per_frame.dtype)  # (B, T).
    denom = logits_len.to(per_frame.dtype).clamp(min=1.0)
    return (per_frame * valid).sum(dim=1) / denom


def _frame_entropy(probs):
    """Per-frame Shannon entropy (nats) of a (B, T, K) distribution.

    Uses torch.xlogy, which returns exactly 0 at p == 0 rather than the
    NaN that `p * log(p)` would produce there. That matters here because
    the smoothed targets are deliberately zero off the active set.
    """
    return -torch.xlogy(probs, probs).sum(dim=-1)


# Per-example diagnostics from the most recent entropy-matched solve.
#
# The solved alpha is not a hyperparameter anyone chose, so the only way
# to know what the method actually did during a run is to record it as it
# happens. That cannot be recovered afterwards: `save_steps` equals
# `max_steps`, so a finished run leaves exactly one checkpoint, and
# probing it measures a single point rather than a trajectory. Worse, a
# checkpoint from a DIFFERENT variant's run answers "what would matching
# pick for this model" rather than "what did matching actually use",
# which is a different question once the smoothing feeds back into the
# model it is measured on.
#
# Populated unconditionally, since every value stored here is already
# computed by the solve -- keeping them costs nothing. Tensors are kept
# on device and left unconverted so that recording never forces a
# GPU->CPU sync in the training loop; the consumer converts them at
# whatever cadence it logs.
_LAST_ENTROPY_MATCH_STATS = {}


def pop_last_entropy_match_stats():
    """Returns and clears diagnostics from the last entropy-matched solve.

    Returns:
        Dict of (B,) tensors -- alpha, h_lo, h_ceiling, h_target, peak --
        or an empty dict if no solve has run since the last call.
    """
    stats = dict(_LAST_ENTROPY_MATCH_STATS)
    _LAST_ENTROPY_MATCH_STATS.clear()
    return stats


def _mixture_entropy(q_tilde, mix, alpha, logits_len):
    """mean_t H((1-a) q~ + a m) per example; `alpha` is (B,)."""
    a = alpha.view(-1, 1, 1)
    mixture = (1.0 - a) * q_tilde + a * mix
    return _masked_frame_mean(_frame_entropy(mixture), logits_len)


def _mixture_entropy_gradient(q_tilde, mix, alpha, logits_len):
    """mean_t h'(a), where h(a) = H((1-a) q~ + a m).

        h'(a) = -sum_c (m_c - q~_c) log z_c(a)

    The +1 from d/dz of -z log z drops out because the coefficients sum
    to zero. Terms where z_c is 0 (off both supports) contribute nothing
    and are masked out rather than evaluated, since log 0 is -inf there.
    """
    a = alpha.view(-1, 1, 1)
    z = (1.0 - a) * q_tilde + a * mix
    support = z > 0
    tiny = torch.finfo(z.dtype).tiny
    terms = (mix - q_tilde) * torch.log(z.clamp(min=tiny))
    per_frame = -torch.where(support, terms, torch.zeros_like(terms)).sum(
        dim=-1)
    return _masked_frame_mean(per_frame, logits_len)


def _entropy_peak_alpha(q_tilde, mix, logits_len, n_iter):
    """Locates a* = argmax_a h(a) on [0, 1].

    Needed only when `mix` is not uniform on the mixture's support. h is
    concave, so h' is non-increasing and a* is where it crosses zero;
    bisecting on h' finds it. When h'(1) >= 0 the function never turns
    over and a* = 1 (the monotone case); when h'(0) <= 0 it is already
    past its peak at a = 0.
    """
    batch = q_tilde.shape[0]
    device = q_tilde.device
    zeros = torch.zeros(batch, device=device)
    ones = torch.ones(batch, device=device)

    grad_at_0 = _mixture_entropy_gradient(q_tilde, mix, zeros, logits_len)
    grad_at_1 = _mixture_entropy_gradient(q_tilde, mix, ones, logits_len)

    lo, hi = zeros.clone(), ones.clone()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        rising = _mixture_entropy_gradient(
            q_tilde, mix, mid, logits_len) > 0.0
        lo = torch.where(rising, mid, lo)
        hi = torch.where(rising, hi, mid)
    peak = 0.5 * (lo + hi)

    peak = torch.where(grad_at_1 >= 0.0, ones, peak)
    peak = torch.where(grad_at_0 <= 0.0, zeros, peak)
    return peak


def _solve_entropy_matched_alpha(q_tilde, mix, h_target, logits_len,
                                  n_iter, monotone):
    """Solves mean_t H((1-a) q~ + a m) = h_target for a, per example.

    Args:
        q_tilde: (B, T, K) target, already restricted to the active set
            and renormalized.
        mix: (B, T, K) distribution being mixed in.
        h_target: (B,) entropy to match.
        logits_len: (B,) valid lengths.
        n_iter: bisection iterations.
        monotone: True when `mix` is uniform on the support, which makes
            h'(1) = 0 and h non-decreasing (see
            apply_entropy_matched_smoothing's docstring). The search then
            runs over the whole of [0, 1]. When False -- the label-space
            case, where the mixing distribution is the blank-heavy
            scatter of a masked uniform -- h can turn over at an interior
            a*, so the peak is located first and the search is confined
            to [0, a*]. Searching [0, 1] there would let bisection land
            on the falling branch, or on nothing at all when both
            endpoints sit below the target.

    Returns:
        (alpha, h_lo, h_ceiling, peak): all (B,). h_ceiling is the
        highest entropy actually reachable, h(a*) -- which is h(1) only
        in the monotone case.
    """
    batch = q_tilde.shape[0]
    device = q_tilde.device
    zeros = torch.zeros(batch, device=device)

    if monotone:
        peak = torch.ones(batch, device=device)
    else:
        peak = _entropy_peak_alpha(q_tilde, mix, logits_len, n_iter)

    h_lo = _mixture_entropy(q_tilde, mix, zeros, logits_len)
    h_ceiling = _mixture_entropy(q_tilde, mix, peak, logits_len)

    lo, hi = zeros.clone(), peak.clone()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        below = _mixture_entropy(q_tilde, mix, mid, logits_len) < h_target
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    alpha = 0.5 * (lo + hi)

    # Cases the search cannot express. Ordering matters: the degenerate
    # check is last so it wins over the ceiling clamp, since when the
    # entropy is flat in alpha any value is equally correct and 0 is the
    # least intrusive.
    reached_ceiling = h_target >= h_ceiling
    already_diffuse = h_target <= h_lo
    degenerate = (h_ceiling - h_lo) < 1e-9
    alpha = torch.where(reached_ceiling, peak, alpha)
    alpha = torch.where(already_diffuse, zeros, alpha)
    alpha = torch.where(degenerate, zeros, alpha)
    return alpha, h_lo, h_ceiling, peak


def _restrict_and_renormalize(probs, eps):
    """Masks to the active set and renormalizes; returns (q~, active, N)."""
    active = probs >= eps
    n_active = active.sum(dim=-1, keepdim=True).clamp(min=1).float()
    restricted = probs * active
    restricted = restricted / restricted.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(torch.float32).tiny)
    return restricted, active, n_active


def apply_entropy_matched_smoothing(est_probs, acoustic_probs, logits_len,
                                     eps=1e-6, n_iter=20, alpha_max=1.0,
                                     kappa=1.0, restrict_reference=False,
                                     return_stats=False):
    """Smooths a target until its entropy matches the model's own.

    This is SETS with beta = 1 (mix in the masked uniform), except that
    alpha is not a hyperparameter: it is SOLVED FOR, once per example,
    so that the smoothed target's mean per-frame entropy equals the
    acoustic posterior's.

    The principle. `est_probs` is the alignment posterior
    p(k_t = c | X, Y): it was computed by conditioning on the ground
    truth Y, so it is more certain than the model's own evidence
    supports, and by exactly the conditional mutual information

        I(K_t ; Y | X) = H(K_t | X) - H(K_t | X, Y)

    Matching the two entropies pays that leaked information back, which
    is why the right amount of smoothing differs per utterance (each one
    leaks a different amount) and why it needs no tuning.

    Concretely, with A = {c : est_probs[c] >= eps}, N = |A|:

        m   = 1/N on A, 0 elsewhere            (masked uniform)
        q~  = est_probs restricted to A, renormalized
        z(a) = (1 - a) * q~ + a * m

    and alpha_b solves, over that example's valid frames,

        mean_t H(z_{b,t}(alpha_b)) = mean_t H(acoustic_probs_{b,t})

    Why bisection is safe here. Writing h(a) = H((1-a) q~ + a m) for one
    frame, H is strictly concave and z(a) is affine in a, so h is
    concave. Its derivative is

        h'(a) = -sum_c (m_c - q~_c) log z_c(a)

    (the +1 from d/dz of -z log z drops out because sum_c (m_c - q~_c) = 0).
    Since m is UNIFORM on A, log m_c = -log N is constant there, so

        h'(1) = log N * sum_c (m_c - q~_c) = 0

    and concavity makes h' non-increasing, hence h'(a) >= h'(1) = 0 on
    [0, 1]. So h -- and its mean over frames -- is non-decreasing, the
    root is unique, and plain bisection converges. This is why q~ is
    renormalized onto A first: with est_probs' sub-eps tail left in, m
    is no longer uniform on the mixture's support, h'(1) is no longer
    exactly 0, and monotonicity degrades from a theorem to an empirical
    observation. (The discarded mass is at most K * eps, ~3e-5 for
    K = 32 and eps = 1e-6.)

    The same reasoning also shows why this function is class-space only:
    the entropy being matched is H(K_t | X) over output CLASSES, so
    est_probs must be over the same alphabet as acoustic_probs. It also
    shows what breaks in label space, where the effective mixing
    distribution is the blank-heavy scatter of the masked uniform rather
    than a uniform: there h'(1) = H(m) - H(q~) - KL(q~ || m) can be
    negative, h peaks at an interior a* and falls afterwards, and
    bisection over [0, 1] would silently return a wrong root. Supporting
    that needs a bracket-then-bisect search, which is deliberately not
    implemented here.

    Args:
        est_probs: Float tensor of shape (B, T, C). The target to smooth,
            i.e. the alignment posterior already mapped into class space.
        acoustic_probs: Float tensor of shape (B, T, C). The model's own
            softmax output, p(k_t | X). Must be over the SAME class axis
            as est_probs. Should be detached: this whole computation is
            target construction, not part of the differentiated graph.
        logits_len: Long tensor of shape (B,). Valid (unpadded) length of
            each sequence.
        eps: Activity threshold. A class counts as reachable at (b, t)
            when est_probs >= eps. This is load-bearing: N = |A| sets the
            highest entropy the mixture can reach (log N), so a larger
            eps lowers that ceiling and makes the alpha = 1 clamp fire
            more often.
        n_iter: Bisection iterations. 20 gives alpha to ~1e-6, far finer
            than needed; the cost is negligible (see below) so there is
            no reason to economize.
        alpha_max: Upper clamp on the solved alpha. 1.0 disables it. A
            safety net for early training, where the model is near-
            uniform and the entropy target can exceed anything the
            mixture can reach.
        kappa: Fraction of the entropy gap to close, in (0, 1]. 1.0 is
            exact matching. Lower values repay only part of the leaked
            information, which keeps the interpretation intact while
            weakening the intervention -- a more principled dial than a
            hard alpha_max if the solved alpha turns out too large.
        return_stats: If True, also returns a dict of diagnostics (see
            Returns). Intended for observe-only pilot runs, where alpha
            is measured but not used.

    Returns:
        Float tensor of shape (B, T, C) with smoothing applied and
        padded time steps zeroed. If return_stats is True, returns
        (smoothed, stats) where stats holds per-example (B,) tensors:
            alpha       solved (and clamped) smoothing weight
            h_lo        mean_t H(q~), the entropy at alpha = 0
            h_hi        mean_t log N, the ceiling at alpha = 1
            h_target    mean_t H(acoustic_probs), the matching target
            case_b      1.0 where h_target >= h_hi (clamped to alpha_max)
            case_c      1.0 where h_target <= h_lo (clamped to 0)
            n_active    mean_t N
            h_prime_1   mean_t h'(1); should be ~0, which empirically
                        checks the monotonicity argument above

    Cost: each bisection step is one (B, T, C) elementwise pass plus a
    reduction, so ~22 passes total. Against a measured 1.74 s training
    step (1hr profile, ~20k frames per batch, C = 32) that is under
    0.05% -- about 1.4 s added over a 2000-step run. Bisection is
    preferred over a fixed 0.01 grid not for speed but for resolution: a
    0.01 grid quantizes a solved alpha of ~0.02 down to three distinct
    levels, which would erase the per-example adaptivity that is the
    whole point of the method.
    """
    assert est_probs.shape == acoustic_probs.shape, (
        f"est_probs {tuple(est_probs.shape)} and acoustic_probs "
        f"{tuple(acoustic_probs.shape)} must share the class axis; this "
        f"smoothing is class-space only (see the docstring).")
    assert 0.0 < kappa <= 1.0, f"kappa must be in (0, 1], got {kappa}"

    target_probs = est_probs.float()
    active = target_probs >= eps
    restricted, _, n_active = _restrict_and_renormalize(target_probs, eps)
    masked_uniform = active.float() / n_active

    reference = acoustic_probs.float()
    if restrict_reference:
        # Measure the model's uncertainty on the SAME support the target
        # lives on, instead of over the whole vocabulary.
        #
        # Without this the two entropies are over different alphabets:
        # H(q~) ranges over the ~3 classes reachable at this frame, while
        # H(p) ranges over all C of them. H(p) is then structurally the
        # larger of the two -- at initialization it is near log C = 3.47
        # against a ceiling of log N ~ 1.2 -- so the target entropy is
        # simply unreachable, alpha clamps at its maximum for the whole
        # run, the model never learns, H(p) never falls, and the clamp
        # never releases. That is the collapse measured at alpha_max=1.0
        # (eval_wer 0.9974 against a 0.2261 baseline).
        #
        # Restricting p to the active set makes H(p|A) <= log N hold by
        # construction, since the uniform is the maximum-entropy
        # distribution on A. Unreachability becomes structurally
        # impossible. Measured on 1hr checkpoints, this also drops the
        # trained-alpha -> solved-alpha slope from ~0.95 (an identity map
        # that returns whatever alpha it was given, hence uninformative)
        # to ~0.75, so the solved alpha actually moves.
        reference = reference * active
        reference = reference / reference.sum(
            dim=-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny)

    return _finish_entropy_matched_smoothing(
        target_probs, restricted, masked_uniform, reference,
        logits_len, est_probs.dtype, n_iter, alpha_max, kappa,
        n_active, monotone=True, return_stats=return_stats)


def apply_label_space_entropy_matched_smoothing(
        est_probs, mix_probs, acoustic_probs, logits_len, n_iter=20,
        alpha_max=1.0, kappa=1.0, return_stats=False):
    """Entropy matching against an arbitrary, non-uniform mixing prior.

    This is the L-SETS-H entry point. The caller supplies both operands
    already scattered into class space:

        est_probs  = scatter(gamma restricted to its active label
                     positions, renormalized)
        mix_probs  = scatter(masked uniform over those label positions)

    Doing it that way is exact rather than an approximation: scatter is
    linear, so smoothing in label space and then scattering equals
    scattering both operands and mixing in class space,

        scatter((1-a) g~ + a m_L) = (1-a) scatter(g~) + a scatter(m_L)

    which is what lets the entropy be evaluated over classes -- the only
    alphabet on which comparing against the acoustic posterior means
    anything -- while the intervention itself is still the label-space
    one.

    The difference from the class-space variant is that `mix_probs` is
    NOT uniform. Roughly half of the blank-augmented label positions are
    blank, so the scattered masked uniform puts ~0.5 on blank and spreads
    the rest by how many reachable positions each class occupies. That
    costs the monotonicity theorem:

        h'(1) = H(m) - H(q~) - KL(q~ || m)

    which is 0 only when m is uniform on the support, and is comfortably
    negative otherwise (~ -1.1 for a confident model against a
    blank-heavy prior). h then rises to an interior peak a* and falls
    after it, so this path locates a* first and searches only [0, a*].
    Consequences worth knowing when reading results from it: the
    reachable entropy ceiling is h(a*), strictly below log N, so the
    ceiling clamp fires more often here than in class space.

    Args:
        est_probs: (B, T, C) scattered, restricted, renormalized target.
        mix_probs: (B, T, C) scattered masked uniform. Must already be a
            distribution over the class axis.
        acoustic_probs: (B, T, C) the model's softmax output, detached.
        logits_len: (B,) valid lengths.
        n_iter: bisection iterations, used for both the peak search and
            the entropy solve.
        alpha_max: upper clamp on the solved alpha.
        kappa: fraction of the entropy gap to close, in (0, 1].
        return_stats: as in apply_entropy_matched_smoothing, with
            `h_hi` reporting h(a*) and an extra `peak` entry holding a*.

    Returns:
        (B, T, C) smoothed target, or (smoothed, stats).
    """
    assert est_probs.shape == acoustic_probs.shape == mix_probs.shape, (
        "est_probs, mix_probs and acoustic_probs must all share the "
        "class axis")
    assert 0.0 < kappa <= 1.0, f"kappa must be in (0, 1], got {kappa}"

    target_probs = est_probs.float()
    n_active = (mix_probs > 0).float().sum(dim=-1, keepdim=True)
    return _finish_entropy_matched_smoothing(
        target_probs, target_probs, mix_probs.float(),
        acoustic_probs.float(), logits_len, est_probs.dtype, n_iter,
        alpha_max, kappa, n_active, monotone=False,
        return_stats=return_stats)


def _finish_entropy_matched_smoothing(original, q_tilde, mix, model_probs,
                                       logits_len, out_dtype, n_iter,
                                       alpha_max, kappa, n_active,
                                       monotone, return_stats):
    """Shared tail of both entropy-matched variants.

    Kept in one place because everything after "what exactly is being
    mixed in" -- the solve, the clamps, the padding mask, the stats --
    is identical whether the prior is uniform or not.
    """
    h_target = _masked_frame_mean(_frame_entropy(model_probs), logits_len)
    h_lo_probe = _mixture_entropy(
        q_tilde, mix, torch.zeros(q_tilde.shape[0], device=q_tilde.device),
        logits_len)
    if kappa < 1.0:
        h_target = h_lo_probe + kappa * (h_target - h_lo_probe)

    alpha, h_lo, h_ceiling, peak = _solve_entropy_matched_alpha(
        q_tilde, mix, h_target, logits_len, n_iter, monotone)
    alpha = alpha.clamp(max=alpha_max)

    _LAST_ENTROPY_MATCH_STATS.clear()
    _LAST_ENTROPY_MATCH_STATS.update({
        "alpha": alpha.detach(),
        "h_lo": h_lo.detach(),
        "h_ceiling": h_ceiling.detach(),
        "h_target": h_target.detach(),
        "peak": peak.detach(),
    })

    a = alpha.view(-1, 1, 1)
    smoothed = (1.0 - a) * q_tilde + a * mix
    # alpha == 0 must be the exact identity: q_tilde can differ from the
    # original by the discarded sub-eps tail.
    smoothed = torch.where(a == 0.0, original, smoothed)

    max_time = original.shape[1]
    time_idx = torch.arange(max_time, device=original.device)
    valid = time_idx.unsqueeze(0) < logits_len.unsqueeze(1)  # (B, T).
    smoothed = (smoothed * valid.unsqueeze(-1).float()).to(out_dtype)

    if not return_stats:
        return smoothed

    stats = {
        "alpha": alpha,
        "h_lo": h_lo,
        "h_hi": h_ceiling,
        "h_target": h_target,
        "case_b": (h_target >= h_ceiling).float(),
        "case_c": (h_target <= h_lo).float(),
        "n_active": _masked_frame_mean(n_active.squeeze(-1), logits_len),
        "h_prime_1": _mixture_entropy_gradient(
            q_tilde, mix,
            torch.ones(q_tilde.shape[0], device=q_tilde.device),
            logits_len),
        "peak": peak,
    }
    return smoothed, stats


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
