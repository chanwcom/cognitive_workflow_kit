"""RNN-Transducer counterpart of `shc_loss.ShcLoss`.

The point of this module is to let the SAME target-smoothing family that
`shc_loss` applies to CTC be applied to an RNN-T model, so that a result
obtained with CTC can be checked for transfer without also changing the
smoothing definition. It therefore reuses `shc_loss_util`'s smoothing
functions verbatim rather than re-deriving them.

What carries over and what does not
-----------------------------------
CTC's alignment variable q_t is "which blank-augmented label position is
frame t aligned to", so its posterior gamma has shape (B, T, L) and
normalizes over the label axis at each frame. `ShcLoss` scatters that into
class space, (B, T, C), and smooths there.

RNN-T's lattice is two-dimensional: a node (t, u) means "t frames
consumed, u labels emitted", and from it the model either emits blank
(advancing t) or emits y_{u+1} (advancing u). The local decision at a node
is therefore a distribution over exactly TWO classes, {blank, y_{u+1}},
and the node's importance is its occupancy posterior gamma(t, u). The
per-node target distribution over the C output classes is

    target[b, t, u, k] = q_blank(t,u)/gamma(t,u)  if k == blank
                         q_label(t,u)/gamma(t,u)  if k == y_{u+1}
                         0                        otherwise

which is the direct analogue of `ShcLoss`'s class-space
`ground_truth_prob`, just indexed by node instead of by frame. Smoothing
then runs on exactly the same object -- a distribution over C classes with
a small active set -- so `shc_loss_util.apply_*_smoothing` applies without
modification.

The active set has size 2 by construction here, whereas in CTC it is
data-dependent (measured ~2.9 late in training with K = 32). Those are
close enough that an alpha tuned on CTC transfers without rescaling, which
is the whole reason for keeping the (alpha, beta) parameterization
identical: alpha is a per-class rate in units of 1/K and does not depend on
the active-set size, so the smoothing MASS lands at alpha*N_a/K in both
cases.

Gradient
--------
For a softmax output at each node,

    d(-log P)/d logit_k(t,u) = gamma(t,u) * ( p(k|t,u) - target(k|t,u) )

so the tail of this file is the same "exp -> subtract -> mask" chain as
`ShcLoss`, with the extra gamma(t,u) weight that CTC does not need (there,
the per-frame occupancy is exactly 1).

Unlike CTC there is no feasibility constraint between T and U: labels are
emitted without consuming frames, so every sample with T >= 1 has at least
one valid alignment and no `valid_sample_mask` analogue is required.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from typing import Optional, Tuple

import torch

from cwk.loss.pytorch import shc_loss_util

# Structurally impossible cells are set to true -inf here, NOT to
# `shc_loss`'s finite LOG_0 = -706.89.
#
# CTC needs the finite floor because its normalization is a logsumexp over
# the whole label axis, and a padded TIME frame masks every position on
# that axis -- an all--inf row would return -inf and the normalization
# would produce NaN. RNN-T never normalizes over a possibly-empty axis:
# the per-node target normalizes over exactly two entries, and inside a
# sample's rectangle the blank entry is always live.
#
# The finite floor is also positively harmful here. A path costs about
# (T + U) * log C in log-probability -- at T = 400, U = 30, C = 32 that is
# -1490, far BELOW -706.89 -- so a floored cell looks *more* probable than
# a real one and wins the logaddexp, which is the RNN-T form of the CTC
# padding bug in CLAUDE.md. `test_long_sequence_does_not_collapse_to_all_blank`
# measured the label-transition mass at inf before this changed.
NEG_INF = float("-inf")

# Kept for callers that still reference the CTC-side constant.
LOG_0 = -706.89


def _skew(x: torch.Tensor, num_diag: int) -> torch.Tensor:
    """Re-indexes a (B, T, U1) lattice tensor to (B, D, U1) diagonals.

    Writes entry (t, u) at (d, u) with d = t + u. In those coordinates both
    of the RNN-T recursions read only from diagonal d-1 (or d+1), so each
    diagonal is fully parallel and the recursion costs T + U sequential
    steps instead of T * U. Out-of-range cells hold LOG_0.

    Args:
        x: (B, T, U1) tensor in lattice coordinates.
        num_diag: D = T + U1 - 1, the number of anti-diagonals.

    Returns:
        (B, D, U1) tensor in diagonal coordinates.
    """
    b, t_len, u1 = x.shape
    u_idx = torch.arange(u1, device=x.device)
    d_idx = torch.arange(num_diag, device=x.device)
    # t = d - u; valid only where 0 <= t < T.
    t_from = d_idx.unsqueeze(1) - u_idx.unsqueeze(0)          # (D, U1)
    valid = (t_from >= 0) & (t_from < t_len)
    gather_t = t_from.clamp(0, t_len - 1).unsqueeze(0).expand(b, -1, -1)
    out = torch.gather(
        x.unsqueeze(1).expand(b, num_diag, t_len, u1),
        2, gather_t.unsqueeze(2)).squeeze(2)
    return out.masked_fill(~valid.unsqueeze(0), NEG_INF)


def _unskew(x: torch.Tensor, t_len: int) -> torch.Tensor:
    """Inverse of `_skew`: (B, D, U1) diagonals back to (B, T, U1)."""
    b, num_diag, u1 = x.shape
    u_idx = torch.arange(u1, device=x.device)
    t_idx = torch.arange(t_len, device=x.device)
    d_from = t_idx.unsqueeze(1) + u_idx.unsqueeze(0)          # (T, U1)
    valid = d_from < num_diag
    gather_d = d_from.clamp(0, num_diag - 1).unsqueeze(0).expand(b, -1, -1)
    out = torch.gather(
        x.unsqueeze(1).expand(b, t_len, num_diag, u1),
        2, gather_d.unsqueeze(2)).squeeze(2)
    return out.masked_fill(~valid.unsqueeze(0), NEG_INF)


def calculate_rnnt_alpha_beta(
        log_p_blank: torch.Tensor,
        log_p_label: torch.Tensor,
        logits_len: torch.Tensor,
        target_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RNN-T forward-backward, the analogue of `shc_loss.calculate_alpha_beta`.

    Recursions (log domain), with u = 0..U indexing labels already emitted:

        alpha(t, u) = ln[ exp(alpha(t-1, u) + b(t-1, u))
                          + exp(alpha(t, u-1) + l(t, u-1)) ]
        beta(t, u)  = ln[ exp(beta(t+1, u) + b(t, u))
                          + exp(beta(t, u+1) + l(t, u)) ]

    with alpha(0, 0) = 0 and beta(T-1, U) = b(T-1, U); then
    log P = alpha(T-1, U) + b(T-1, U) = beta(0, 0).

    Both are run in diagonal coordinates (see `_skew`) so that the whole
    lattice costs T + U sequential steps.

    Precision is promoted to at least float32 -- in bf16 the T + U
    accumulated logaddexp's drift far enough to break the agreement with
    torchaudio -- but a float64 input is kept in float64. That matters for
    long lattices: at T = 400 the float32 recursion leaves the total blank
    transition mass at 400.21 instead of 400 (5e-4 relative), which is
    harmless for a gradient but too loose to use as a correctness check,
    so the tests exercise the float64 path when they assert on it.

    Args:
        log_p_blank: (B, T, U+1) log p(blank | t, u).
        log_p_label: (B, T, U+1) log p(y_{u+1} | t, u). Column U is unused
            (no label remains to emit) and may hold anything.
        logits_len: (B,) valid frame count per sample.
        target_lens: (B,) label count U_b per sample.

    Returns:
        (log_alpha, log_beta, log_seq_prob), the first two of shape
        (B, T, U+1) in lattice coordinates and the last of shape (B,).
    """
    dtype_in = log_p_blank.dtype
    work_dtype = (torch.float64 if dtype_in == torch.float64
                  else torch.float32)
    log_p_blank = log_p_blank.to(work_dtype)
    log_p_label = log_p_label.to(work_dtype)
    b, t_len, u1 = log_p_blank.shape
    device = log_p_blank.device
    num_diag = t_len + u1 - 1

    # Cells outside a sample's own (T_b, U_b) rectangle must not carry
    # probability. Masking the per-node log-probs once here is enough: a
    # blocked cell can then never be entered, so neither recursion needs a
    # per-sample bound inside its loop.
    t_ar = torch.arange(t_len, device=device).view(1, -1, 1)
    u_ar = torch.arange(u1, device=device).view(1, 1, -1)
    in_t = t_ar < logits_len.view(-1, 1, 1)
    in_u = u_ar <= target_lens.view(-1, 1, 1)
    node_ok = in_t & in_u
    # Emitting a label is only possible while labels remain (u < U_b).
    label_ok = node_ok & (u_ar < target_lens.view(-1, 1, 1))
    log_p_blank = log_p_blank.masked_fill(~node_ok, NEG_INF)
    log_p_label = log_p_label.masked_fill(~label_ok, NEG_INF)

    b_sk = _skew(log_p_blank, num_diag)
    l_sk = _skew(log_p_label, num_diag)

    neg = torch.full((b, u1), NEG_INF, device=device, dtype=work_dtype)

    # ---- forward ----
    alpha = torch.full((b, num_diag, u1), NEG_INF, device=device,
                       dtype=work_dtype)
    prev = neg.clone()
    prev[:, 0] = 0.0                      # alpha(0, 0) = log 1
    alpha[:, 0] = prev
    for d in range(1, num_diag):
        # From (t-1, u): same u on diagonal d-1.
        from_blank = prev + b_sk[:, d - 1]
        # From (t, u-1): u shifted by one on diagonal d-1.
        shifted = torch.cat([neg[:, :1], (prev + l_sk[:, d - 1])[:, :-1]],
                            dim=1)
        prev = torch.logaddexp(from_blank, shifted)
        alpha[:, d] = prev

    # ---- backward ----
    beta = torch.full((b, num_diag, u1), NEG_INF, device=device,
                      dtype=work_dtype)
    # beta at the terminal node (T_b - 1, U_b) is its blank log-prob; every
    # other cell of the last diagonal is unreachable. The terminal node's
    # diagonal is d = T_b - 1 + U_b, which differs per sample, so the
    # initialization is scattered rather than written into `num_diag - 1`.
    term_d = (logits_len - 1 + target_lens).clamp(min=0)
    nxt = neg.clone()
    for d in range(num_diag - 1, -1, -1):
        is_term = (term_d == d)
        if bool(is_term.any()):
            # Seed the terminal node for the samples whose last diagonal
            # this is, before the recursion consumes `nxt`.
            seed = torch.full_like(nxt, NEG_INF)
            idx = torch.nonzero(is_term, as_tuple=True)[0]
            seed[idx, target_lens[idx]] = b_sk[idx, d, target_lens[idx]]
            nxt = torch.where(is_term.view(-1, 1), seed, nxt)
            beta[:, d] = torch.where(is_term.view(-1, 1), nxt, beta[:, d])
        if d == 0:
            break
        # beta(t, u) reads beta(t+1, u) (same u, diagonal d) and
        # beta(t, u+1) (u+1, diagonal d). Written for diagonal d-1.
        cur = beta[:, d] if not bool(is_term.any()) else beta[:, d]
        to_blank = cur + b_sk[:, d - 1]
        up = torch.cat([cur[:, 1:], neg[:, :1]], dim=1) + l_sk[:, d - 1]
        combined = torch.logaddexp(to_blank, up)
        # Do not overwrite a terminal seed already placed on d-1.
        beta[:, d - 1] = torch.where(
            (term_d == d - 1).view(-1, 1),
            beta[:, d - 1], combined)
        nxt = beta[:, d - 1]

    log_alpha = _unskew(alpha, t_len)
    log_beta = _unskew(beta, t_len)
    log_seq_prob = log_beta[:, 0, 0]

    return (log_alpha.to(dtype_in), log_beta.to(dtype_in),
            log_seq_prob.to(dtype_in))


def rnnt_transition_posteriors(
        log_alpha: torch.Tensor,
        log_beta: torch.Tensor,
        log_p_blank: torch.Tensor,
        log_p_label: torch.Tensor,
        log_seq_prob: torch.Tensor,
        logits_len: torch.Tensor,
        target_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Posterior of taking each of the two transitions out of node (t, u).

        q_blank(t,u) = alpha(t,u) p_blank(t,u) beta(t+1,u)   / P
        q_label(t,u) = alpha(t,u) p_label(t,u) beta(t,u+1)   / P

    Their sum is the node occupancy gamma(t,u) = alpha(t,u) beta(t,u) / P,
    which is what weights the node's gradient contribution.

    Two things here are easy to get wrong and both are pinned by tests:

    The TERMINAL transition leaves the lattice. beta(t, u) is defined as
    the log-probability of everything still to come from node (t, u), so
    beta(T_b - 1, U_b) already equals that node's own blank log-prob and
    the "beta(t+1, u)" factor for it is log 1, not log 0. Reading it out of
    the shifted tensor instead gives LOG_0 there, which drops exactly one
    unit of transition mass -- `test_occupancy_sums_to_path_length` sees
    the total come out at T + U - 1.

    Cells OUTSIDE a sample's (T_b, U_b) rectangle must be forced to zero
    rather than left to the arithmetic. alpha and beta are floored at the
    finite LOG_0 there, so alpha + beta - log P evaluates to
    2*LOG_0 - log P, which is POSITIVE (and exp overflows) as soon as
    log P drops below 2*LOG_0 -- i.e. for any long, not-yet-confident
    sample. This is the RNN-T form of the CTC padding bug recorded in
    CLAUDE.md, and `test_long_sequence_does_not_collapse_to_all_blank`
    fires on it: the label mass came out inf.

    Returns:
        (q_blank, q_label), both (B, T, U+1) in PROBABILITY domain, exactly
        zero outside each sample's valid rectangle.
    """
    b, t_len, u1 = log_alpha.shape
    device = log_alpha.device
    neg = torch.full((b, 1, u1), NEG_INF, device=device, dtype=log_alpha.dtype)
    beta_t1 = torch.cat([log_beta[:, 1:], neg], dim=1)            # beta(t+1,u)
    neg_u = torch.full((b, t_len, 1), NEG_INF, device=device,
                       dtype=log_alpha.dtype)
    beta_u1 = torch.cat([log_beta[:, :, 1:], neg_u], dim=2)       # beta(t,u+1)

    t_ar = torch.arange(t_len, device=device).view(1, -1, 1)
    u_ar = torch.arange(u1, device=device).view(1, 1, -1)
    node_ok = (t_ar < logits_len.view(-1, 1, 1)) & (
        u_ar <= target_lens.view(-1, 1, 1))
    label_ok = node_ok & (u_ar < target_lens.view(-1, 1, 1))
    is_exit = (t_ar == (logits_len - 1).view(-1, 1, 1)) & (
        u_ar == target_lens.view(-1, 1, 1))
    beta_t1 = beta_t1.masked_fill(is_exit, 0.0)

    lp = log_seq_prob.view(-1, 1, 1)
    log_qb = (log_alpha + log_p_blank + beta_t1 - lp).masked_fill(
        ~node_ok, float("-inf"))
    log_ql = (log_alpha + log_p_label + beta_u1 - lp).masked_fill(
        ~label_ok, float("-inf"))
    return log_qb.exp(), log_ql.exp()


def _node_target(q_blank: torch.Tensor, q_label: torch.Tensor,
                 labels: torch.Tensor, num_classes: int,
                 blank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds the per-node class-space target and the node occupancy.

    The returned target is NORMALIZED over classes at every node, so it is
    the same kind of object `ShcLoss` hands to the smoothing functions --
    p(k | node) rather than an unnormalized posterior mass. The occupancy
    is kept separately and multiplied back in only at the gradient, which
    is what lets the smoothing be defined identically to the CTC case.

    Args:
        q_blank: (B, T, U+1) probability of the blank transition.
        q_label: (B, T, U+1) probability of the label transition.
        labels: (B, U) label ids (no blanks inserted).
        num_classes: C.
        blank: blank class id.

    Returns:
        (target, gamma) of shapes (B, T, U+1, C) and (B, T, U+1).
    """
    b, t_len, u1 = q_blank.shape
    gamma = q_blank + q_label
    denom = gamma.clamp(min=1e-30)
    target = q_blank.new_zeros((b, t_len, u1, num_classes))
    target[..., blank] = q_blank / denom
    # Column u of `target` emits y_{u+1}, i.e. labels[:, u]; the last
    # column has no label left and keeps only its blank entry.
    lab = labels.clamp(min=0)
    lab_idx = lab.view(b, 1, -1, 1).expand(b, t_len, lab.shape[1], 1)
    target[:, :, :lab.shape[1], :].scatter_add_(
        3, lab_idx, (q_label[:, :, :lab.shape[1]] /
                     denom[:, :, :lab.shape[1]]).unsqueeze(3))
    return target, gamma


def apply_frame_label_support_smoothing(
        target: torch.Tensor,
        gamma: torch.Tensor,
        blank: int,
        alpha: float,
        beta: float,
        eps: float = 1e-10,
) -> torch.Tensor:
    """FAS confined to the label subspace, with a FRAME-level active set.

    Why this exists, and why plain FAS is not the right transfer of the
    CTC result to RNN-T.

    A node's target has exactly two non-zero entries, {blank, y_{u+1}}, so
    `apply_floored_active_support_smoothing` sees N_a = 2 there. Its change
    to the blank entry is

        d target(blank) = (alpha/K) * (1 - N_a * q_b)

    which is +alpha/K wherever q_b -> 0, i.e. at exactly the nodes where a
    label has to be emitted -- and that is the only chance to emit it,
    since RNN-T emits each label once. Measured on a converged 100 h model
    (alpha = 0.05, seed 0, full dev): deletions +18 %, insertions -28 %,
    substitutions unchanged. The operating point moved along the D/I
    trade-off without any gain in discrimination.

    Note the blank pressure does NOT come from N_a being small: the limit
    above is independent of N_a, so widening the active set cannot remove
    it (and makes the true label lose more mass, not less). Excluding
    blank from a PER-NODE active set is no use either -- one class would be
    left and the result is a deterministic blank penalty, not smoothing.

    What CTC actually has that RNN-T does not is redundancy: a label spans
    several frames there, so no single frame's argmax is decisive, and the
    frame's active set holds blank plus several DISTINCT labels. The
    transfer that preserves that structure is to leave the blank/label
    split alone and smooth only WITHIN the label part, over the labels the
    lattice could emit at this frame:

        A_t = {c != blank : pbar_t(c) > eps},
        pbar_t = sum_u gamma(t,u) target(t,u,.) / sum_u gamma(t,u)

    the occupancy-weighted marginal over the u axis -- the same object CTC
    smooths. Measured |A_t| on the model above: 2.48 labels at eps = 1e-10,
    against CTC's 2.9, so the set is genuinely multi-class. It collapses
    below ~1 by eps = 1e-2, so the small default matters here.

    Writing s_b = q_blank/gamma and s = q_label/gamma = 1 - s_b, the node
    target becomes

        target(blank) = s_b                                    (unchanged)
        target(c)     = s * [(1 - m) e_{y_{u+1}}(c) + r(c)]    (c != blank)

    with active rate alpha/K as in FAS, r(c) = (alpha/K)(1_{A_t}(c) +
    beta (1 - 1_{A_t}(c))) and m = (alpha/K)(N_a + beta(K - 1 - N_a)), so
    the label part keeps its total mass s and d target(blank) = 0 exactly.
    beta keeps its FAS meaning: 0 restricts smoothing to A_t, 1 is uniform
    LS over the non-blank classes.

    Args:
        target: (B, T, U+1, C) per-node class-space target, normalized over
            C, as `_node_target` returns it.
        gamma: (B, T, U+1) node occupancy, exactly zero outside each
            sample's valid rectangle.
        blank: blank class id.
        alpha: smoothing strength, a per-class rate in 1/C as everywhere
            else in this family.
        beta: floor height for the inactive labels.
        eps: activity threshold on the frame marginal.

    Returns:
        (B, T, U+1, C) smoothed target. Nodes with no label mass (the u = U
        column, and everything outside the rectangle) are returned
        untouched.
    """
    b, t_len, u1, num_classes = target.shape
    dt = target.dtype
    occ = gamma.sum(2).clamp(min=1e-30)                             # (B,T)
    pbar = (gamma.unsqueeze(3) * target).sum(2) / occ.unsqueeze(2)  # (B,T,C)

    active = pbar > eps
    active[..., blank] = False
    n_a = active.sum(-1, keepdim=True).to(dt)                       # (B,T,1)
    k = float(num_classes)
    rate = alpha / k
    mass = rate * (n_a + beta * (k - 1.0 - n_a))                    # (B,T,1)

    floor = rate * (active.to(dt) + beta * (~active).to(dt))        # (B,T,C)
    floor[..., blank] = 0.0

    s_b = target[..., blank]                                        # (B,T,U+1)
    label_part = target.clone()
    label_part[..., blank] = 0.0
    s = label_part.sum(-1)                                          # (B,T,U+1)
    # The conditional is only defined where there is label mass; where
    # there is none the whole label part stays zero, which is what
    # multiplying by s does anyway.
    cond = label_part / s.clamp(min=1e-30).unsqueeze(3)
    new_cond = (1.0 - mass).unsqueeze(2) * cond + floor.unsqueeze(2)
    out = new_cond * s.unsqueeze(3)
    out[..., blank] = s_b
    return out


def diagonal_token_mass(target: torch.Tensor,
                        gamma: torch.Tensor) -> torch.Tensor:
    """Token mass crossing each anti-diagonal of the lattice.

    k = t + u indexes the STEP of an alignment path: every transition
    advances t or u by exactly one, so a path visits exactly one node per
    anti-diagonal and

        sum_{t+u=k} gamma(t, u) = 1

    for every k. Measured on the converged 100 h model: mean deviation
    5.0e-6, max 2.3e-5 over 4815 diagonals, i.e. exact to float32. This is
    the RNN-T analogue of CTC's per-frame normalization sum_l gamma(t,l)=1,
    and the reason a per-FRAME set is the wrong object here: at fixed t a
    path can climb several u, so those nodes are SEQUENTIAL, not
    alternatives, and sum_u gamma(t,u) is not 1.

    gamma(t,u) * target(t,u,c) is the mass of the transition that leaves
    (t,u) emitting c, so summing it along an anti-diagonal gives the
    distribution of "which token is emitted at step k". Its class sum is 1.

    Args:
        target: (B, T, U+1, C) per-node class-space target.
        gamma: (B, T, U+1) node occupancy.

    Returns:
        (B, T + U, C), row k holding the token mass LEAVING diagonal k
        (equivalently, arriving at diagonal k + 1).
    """
    b, t_len, u1, num_classes = target.shape
    device = target.device
    diag = (torch.arange(t_len, device=device).view(-1, 1) +
            torch.arange(u1, device=device).view(1, -1))          # (T, U+1)
    out = target.new_zeros((b, t_len + u1 - 1, num_classes))
    out.index_add_(1, diag.reshape(-1),
                   (gamma.unsqueeze(3) * target).reshape(
                       b, t_len * u1, num_classes))
    return out


def apply_diagonal_active_support_smoothing(
        target: torch.Tensor,
        gamma: torch.Tensor,
        alpha: float,
        beta: float,
        eps: float = 1e-10,
        align: str = "departure",
) -> torch.Tensor:
    """FAS whose active set is shared by every node on an anti-diagonal.

    The per-node active set is {blank, y_{u+1}}, which is why plain FAS
    degenerates on RNN-T (all of the smoothing mass it takes off the true
    label lands on blank). Recovering a multi-class set by marginalizing
    over u at fixed t does NOT work -- measured +4.5 % at alpha = 0.05 and
    +33.4 % at 0.10 against the baseline -- because those nodes lie on one
    path one after another, so the labels pulled in belong to OTHER
    transcript positions (measured: 93.7 % of frames span 2-4 positions),
    and given the node's predictor history they are simply wrong.

    The anti-diagonal is the object that fixes this. A path visits exactly
    one node per diagonal, so the tokens crossing it are genuine competing
    hypotheses for the same step, and their masses form a distribution
    (see `diagonal_token_mass`). Every node on the diagonal therefore
    shares one active set

        A_k = {c : P_k(c) > eps}

    and FAS runs per node against it, exactly as in the CTC case.

    `align` picks which diagonal's transitions a node reads, and the two
    differ by one step:

      "departure" -- the transitions LEAVING the node's own diagonal.
          Self-consistent: the node's target is a distribution over what it
          emits next, which is one of those transitions.
      "arrival" -- the transitions arriving at the node's diagonal, i.e.
          leaving diagonal k-1. This is one step behind the node's own
          decision, and the node's own next label y_{u+1} then has to be
          supplied by a DIFFERENT node on the diagonal; when that node has
          little occupancy it drops out. Measured gamma-weighted, the true
          label is missing from the set 42.7 % of the time under "arrival"
          against 15.2 % under "departure", and a missing true label means
          FAS takes mass off it and returns none.

    Args:
        target: (B, T, U+1, C) per-node class-space target, normalized
            over C, as `_node_target` returns it.
        gamma: (B, T, U+1) node occupancy, zero outside the rectangle.
        alpha: smoothing strength, a per-class rate in 1/C.
        beta: floor height for the inactive classes, as in FAS.
        eps: activity threshold on the diagonal's token mass.
        align: "departure" or "arrival".

    Returns:
        (B, T, U+1, C) smoothed target.
    """
    assert align in ("departure", "arrival"), align
    b, t_len, u1, num_classes = target.shape
    device, dt = target.device, target.dtype
    p_diag = diagonal_token_mass(target, gamma)                   # (B, K, C)
    if align == "arrival":
        # Row k must hold what arrives at k, i.e. what left k - 1.
        p_diag = torch.cat(
            [p_diag.new_zeros((b, 1, num_classes)), p_diag[:, :-1]], dim=1)

    active = p_diag > eps                                          # (B, K, C)
    n_a = active.sum(-1, keepdim=True).to(dt)                      # (B, K, 1)
    k = float(num_classes)
    rate = alpha / k
    mass = rate * (n_a + beta * (k - n_a))                         # (B, K, 1)
    floor = rate * (active.to(dt) + beta * (~active).to(dt))       # (B, K, C)

    diag = (torch.arange(t_len, device=device).view(-1, 1) +
            torch.arange(u1, device=device).view(1, -1)).reshape(-1)
    floor_n = floor.index_select(1, diag).view(b, t_len, u1, num_classes)
    mass_n = mass.index_select(1, diag).view(b, t_len, u1, 1)
    return target * (1.0 - mass_n) + floor_n


class RnntShcLoss(torch.autograd.Function):
    """RNN-T loss with the `shc_loss` target-smoothing family applied.

    Mirrors `shc_loss.ShcLoss`: `forward` runs the forward-backward, builds
    the class-space target, post-processes it with the requested smoothing,
    stashes the resulting gradient, and returns only the loss; `backward`
    scales the stashed gradient by the incoming one.
    """

    @staticmethod
    def forward(ctx,
                labels,
                target_lens,
                logits,
                logits_len,
                blank=0,
                alpha=0.0,
                beta=0.0,
                alpha_mode="fixed",
                fas_eps=1e-10,
                asap_eps=1e-3,
                diag_align="departure"):
        """Calculates the smoothed RNN-T loss.

        Args:
            labels: (B, U) ground-truth label ids WITHOUT blanks. Unlike
                `ShcLoss` this must not be blank-augmented: RNN-T's lattice
                supplies the blanks itself, one per frame.
            target_lens: (B,) number of real labels per sample.
            logits: (B, T, U+1, C) joint-network output.
            logits_len: (B,) valid frame count per sample.
            blank: blank class id (torchaudio's rnnt_loss uses the same
                argument name and defaults it to the caller).
            alpha: smoothing strength, same units as `ShcLoss` (a
                per-class rate in 1/C).
            beta: floor height for "floored_active_support"; 0 restricts
                smoothing to the active set and 1 is textbook uniform LS.
            alpha_mode: "fixed", "active_support",
                "floored_active_support", "frame_label_support" or "asap".
                There is no "label"/"class" smoothing_space choice here --
                the per-node target only exists in class space, so the
                label-axis variants have no analogue.
                "frame_label_support" is the one mode that is not a
                per-node rule; see `apply_frame_label_support_smoothing`
                for why RNN-T needs it where CTC does not.
            fas_eps: activity threshold for floored_active_support,
                frame_label_support and diagonal_active_support.
            diag_align: "departure" or "arrival"; only diagonal_active_support
                reads it. See
                `apply_diagonal_active_support_smoothing`.
            asap_eps: activity threshold for asap.

        Returns:
            (B,) loss, i.e. -log P(y | x) per sample.
        """
        assert labels.dim() == 2, labels.shape
        assert logits.dim() == 4, (
            "RNN-T logits must be (B, T, U+1, C); got "
            f"{tuple(logits.shape)}")
        assert logits.shape[0] == labels.shape[0]
        assert logits.shape[2] == labels.shape[1] + 1, (
            "logits' third axis must be U+1 where U is labels' width; got "
            f"{logits.shape[2]} vs {labels.shape[1]} + 1")
        assert alpha_mode in ("fixed", "active_support",
                              "floored_active_support",
                              "frame_label_support",
                              "diagonal_active_support", "asap"), alpha_mode

        b, t_len, u1, num_classes = logits.shape
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        log_p_blank = log_probs[..., blank]
        lab = labels.clamp(min=0)
        # p(y_{u+1} | t, u) for u < U; the last column is never used by the
        # recursion (label_ok masks it) but must exist to keep shapes.
        gathered = torch.gather(
            log_probs[:, :, :u1 - 1, :], 3,
            lab.view(b, 1, u1 - 1, 1).expand(b, t_len, u1 - 1, 1)).squeeze(3)
        log_p_label = torch.cat(
            [gathered, torch.full((b, t_len, 1), NEG_INF,
                                  device=logits.device)], dim=2)

        log_alpha, log_beta, log_seq_prob = calculate_rnnt_alpha_beta(
            log_p_blank, log_p_label, logits_len, target_lens)
        loss = -log_seq_prob

        q_blank, q_label = rnnt_transition_posteriors(
            log_alpha, log_beta, log_p_blank, log_p_label, log_seq_prob,
            logits_len, target_lens)
        target, gamma = _node_target(q_blank, q_label, lab, num_classes,
                                     blank)

        smoothing_enabled = alpha > 0.0
        if smoothing_enabled and alpha_mode == "diagonal_active_support":
            target = apply_diagonal_active_support_smoothing(
                target, gamma, alpha, beta, eps=fas_eps, align=diag_align)
        elif smoothing_enabled and alpha_mode == "frame_label_support":
            # Needs the (t, u) structure and the occupancy, so it does not
            # go through the flattened per-node path below.
            target = apply_frame_label_support_smoothing(
                target, gamma, blank, alpha, beta, eps=fas_eps)
        elif smoothing_enabled:
            # The smoothing functions take (B, N, C) plus a per-sample
            # valid length along N. Nodes are folded into one axis and the
            # length is set to the full width, because validity here is a
            # 2-D (t, u) rectangle rather than a prefix -- the real masking
            # happens through `gamma`, which is exactly zero outside the
            # rectangle and so zeroes those nodes' gradients regardless.
            flat = target.reshape(b, t_len * u1, num_classes)
            full_len = torch.full((b,), t_len * u1, dtype=torch.long,
                                  device=logits.device)
            if alpha_mode == "active_support":
                flat = shc_loss_util.apply_active_support_smoothing(
                    flat, full_len, alpha)
            elif alpha_mode == "floored_active_support":
                flat = shc_loss_util.apply_floored_active_support_smoothing(
                    flat, full_len, alpha, beta, eps=fas_eps)
            elif alpha_mode == "asap":
                flat = shc_loss_util.apply_active_support_acoustic_smoothing(
                    flat, log_probs.exp().reshape(
                        b, t_len * u1, num_classes), full_len,
                    alpha, beta, eps=asap_eps)
            else:
                flat = shc_loss_util.apply_post_processing(
                    flat, full_len, alpha, beta)
            target = flat.reshape(b, t_len, u1, num_classes)

        # d(-log P)/d logit = gamma * (p - target). gamma is zero outside
        # each sample's (T_b, U_b) rectangle, so no extra mask is needed.
        gradient = gamma.unsqueeze(3) * (log_probs.exp() - target)
        ctx.save_for_backward(gradient.to(logits.dtype))
        return loss.to(logits.dtype)

    @staticmethod
    def backward(ctx, grad):
        gradient, = ctx.saved_tensors
        gradient = gradient * grad.view(-1, 1, 1, 1)
        # One entry per non-ctx argument of `forward`, in order: labels,
        # target_lens, logits, logits_len, blank, alpha, beta, alpha_mode,
        # fas_eps, asap_eps, diag_align. Only `logits` (position 3) gets a
        # gradient.
        return (None, None, gradient, None, None, None, None, None, None,
                None, None)


def rnnt_shc_loss(labels, target_lens, logits, logits_len, blank=0,
                  alpha=0.0, beta=0.0, alpha_mode="fixed",
                  fas_eps=1e-10, asap_eps=1e-3, reduction="mean",
                  diag_align="departure"):
    """Thin functional wrapper, mirroring torchaudio.functional.rnnt_loss."""
    loss = RnntShcLoss.apply(labels, target_lens, logits, logits_len, blank,
                             alpha, beta, alpha_mode, fas_eps, asap_eps,
                             diag_align)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction {reduction!r}")
