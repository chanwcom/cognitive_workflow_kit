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



def apply_alignment_weight_smoothing(gamma, logits_len, target_lens,
                                     alpha, beta, eps=1e-3):
    """AWS: smooths the alignment WEIGHT along each anti-diagonal.

    Alignment Weight Smoothing. Smooths the node occupancy, not the target.

    Every other smoothing in this file rewrites the per-node target
    z_hat(t, u, .), which is where RNN-T differs from CTC and where it
    keeps losing: z_hat has at most two non-zero entries (blank and
    y_{u+1}), so it is a binary distribution already sitting at a
    structural ceiling, and perturbing it in either direction -- flatter
    (`floored_active_support`) or sharper (`sharpen`) -- costs WER
    monotonically.

    gamma is the other factor. The gradient is

        dL/dz(t, u, c) = gamma(t, u) * ( p(c | t, u) - z_hat(t, u, c) )

    so gamma is a pure per-node WEIGHT and z_hat is the target. Smoothing
    gamma therefore leaves every node's target exactly as the lattice
    computed it -- the model is never asked to put mass on a wrong label
    -- and changes only how much each alignment position is trained on.

    The axis is the anti-diagonal t + u = k, because that is the one whose
    occupancy is a probability distribution: every transition raises t + u
    by exactly one, so a path crosses each anti-diagonal exactly once and
    `sum_{t+u=k} gamma(t, u) = 1` (verified to 4e-6). Columns are not
    normalized -- `sum_u gamma(t, u)` is the expected number of nodes
    visited in frame t, which exceeds 1 whenever the model emits more than
    one label in a frame. This is the RNN-T analogue of CTC's
    `sum_l gamma(t, l) = 1`, which is what CTC's smoothing_space="label"
    smooths.

    In a trained model each anti-diagonal is nearly a Viterbi decision --
    one node holds 97-98 % of the mass with 2-3 above `eps` -- so the
    smoothing says "do not be this certain about WHERE on this diagonal
    the path sits".

    Args:
        gamma: (B, T, U+1) node occupancies, exactly zero outside each
            sample's own (T_b, U_b+1) rectangle.
        logits_len: (B,) valid frame count per sample.
        target_lens: (B,) real label count per sample.
        alpha: per-node rate, same units as the class-space modes: the
            floor added to each active node is alpha / K, where K is the
            number of VALID nodes on that diagonal (the analogue of the
            vocabulary size in class space). K varies with k, so the floor
            does too.
        beta: 0 mixes toward uniform over the active nodes only, 1 toward
            uniform over every valid node on the diagonal, linear between.
        eps: a node is active when its occupancy exceeds this.

    Returns:
        (B, T, U+1) smoothed occupancies. Each diagonal still sums to 1,
        and nodes outside the rectangle stay exactly zero.
    """
    b, t_len, u1 = gamma.shape
    device = gamma.device
    n_diag = t_len + u1

    t_idx = torch.arange(t_len, device=device).view(-1, 1)
    u_idx = torch.arange(u1, device=device).view(1, -1)
    diag = (t_idx + u_idx).reshape(-1)                      # (T*(U+1),)

    valid = ((t_idx < logits_len.view(-1, 1, 1))
             & (u_idx <= target_lens.view(-1, 1, 1)))       # (B, T, U+1)
    active = (gamma > eps) & valid

    def per_diag(x):
        """Sums x over each anti-diagonal -> (B, T+U)."""
        out = torch.zeros(b, n_diag, dtype=gamma.dtype, device=device)
        return out.index_add_(1, diag, x.reshape(b, -1).to(gamma.dtype))

    n_active = per_diag(active).clamp(min=1.0)
    n_valid = per_diag(valid).clamp(min=1.0)

    take = lambda d: d.index_select(1, diag).reshape(b, t_len, u1)
    # Exactly `apply_floored_active_support_smoothing`'s two-rate rule,
    # with the diagonal standing in for the class axis: every active node
    # gets alpha/K, every inactive valid one gets beta*alpha/K, and the
    # diagonal is renormalized so it still sums to one. beta = 0 is
    # active-support only, beta = 1 is uniform over the whole diagonal.
    rate = alpha / take(n_valid)
    inactive = valid & ~active
    mixin = rate * (active.to(gamma.dtype) + beta * inactive.to(gamma.dtype))
    # The mass handed out has to come from somewhere, and it comes out of
    # the ACTIVE nodes alone -- scaled by (1 - added/M_A), where M_A is the
    # active nodes' own share of the diagonal. At beta = 0 an inactive node
    # is then left exactly as the lattice computed it.
    #
    # shc_loss_util's class-space FAS instead writes
    # `(1 - mass) * y + mixin`, shrinking active and inactive alike. The
    # two agree whenever the inactive mass is negligible, which in class
    # space at eps = 1e-10 it is (inactive classes sit below 1e-10). On a
    # diagonal at eps = 1e-3 it is not, so the distinction is real here.
    added = take(per_diag(mixin))
    m_active = take(per_diag(gamma * active.to(gamma.dtype))).clamp(min=1e-12)
    scale = (1.0 - added / m_active).clamp(min=0.0)
    smoothed = torch.where(active, gamma * scale + mixin, gamma + mixin)
    # Outside the rectangle `add` is zero and gamma is exactly zero, so
    # those nodes stay zero; the where() makes that explicit.
    return torch.where(valid, smoothed, gamma)



def apply_alpha_beta_diagonal_smoothing(log_x: torch.Tensor,
                                        logits_len: torch.Tensor,
                                        target_lens: torch.Tensor,
                                        alpha: float,
                                        beta: float,
                                        eps: float = 1e-10) -> torch.Tensor:
    """AWS applied to a LOG-domain lattice half (log_alpha or log_beta).

    `apply_alignment_weight_smoothing` smooths gamma, whose anti-diagonals
    are probability distributions (`sum_{t+u=k} gamma = 1`, verified to
    4e-6), so an absolute floor of alpha/K per node is meaningful there.
    log_alpha and log_beta have no such normalization -- their diagonal
    sums span many orders of magnitude -- so each diagonal is first
    rescaled to sum to one, smoothed by exactly the same rule, and then
    put back on its original scale. That makes the smoothing
    scale-invariant, which it has to be: alpha and beta are only defined
    up to the per-diagonal factor that cancels in gamma = alpha*beta/P.

    Smoothing the two halves is not the same intervention twice:

        gamma(t,u)    = alpha(t,u) beta(t,u) / P
        z_hat(t,u,bl) = p(blank|t,u) beta(t+1,u) / beta(t,u)

    alpha cancels out of z_hat entirely (verified: rescaling alpha per
    node moves z_hat by 0.0000 while rescaling beta moves it by 0.1108),
    so smoothing alpha moves only the WEIGHT, and smoothing beta moves the
    weight and the TARGET together.

    Args:
        log_x: (B, T, U+1) log_alpha or log_beta, floored at LOG_0 outside
            each sample's rectangle.
        logits_len: (B,) valid frame count per sample.
        target_lens: (B,) real label count per sample.
        alpha: AWS rate, in units of 1/K with K the diagonal's valid node
            count -- the same units as `apply_alignment_weight_smoothing`.
        beta: AWS floor height for inactive nodes.
        eps: a node is active when its rescaled diagonal share exceeds it.

    Returns:
        (B, T, U+1) smoothed log-lattice, untouched outside the rectangle.
    """
    b, t_len, u1 = log_x.shape
    device = log_x.device
    n_diag = t_len + u1

    t_idx = torch.arange(t_len, device=device).view(-1, 1)
    u_idx = torch.arange(u1, device=device).view(1, -1)
    diag = (t_idx + u_idx).reshape(-1)
    valid = ((t_idx < logits_len.view(-1, 1, 1))
             & (u_idx <= target_lens.view(-1, 1, 1)))

    take = lambda d: d.index_select(1, diag).reshape(b, t_len, u1)

    def per_diag(x):
        out = torch.zeros(b, n_diag, dtype=log_x.dtype, device=device)
        return out.index_add_(1, diag, x.reshape(b, -1).to(log_x.dtype))

    # Rescale each diagonal to sum to one, in a max-shifted exponential so
    # the deep underflow that makes the log domain necessary in the first
    # place does not reappear here.
    masked = log_x.masked_fill(~valid, NEG_INF)
    peak = torch.full((b, n_diag), NEG_INF, dtype=log_x.dtype, device=device)
    peak = peak.index_reduce_(1, diag, masked.reshape(b, -1), "amax",
                              include_self=True)
    live = torch.isfinite(peak)
    peak = torch.where(live, peak, torch.zeros_like(peak))
    share = (masked - take(peak)).exp()                 # 0 where invalid
    total = per_diag(share)
    share = share / take(total).clamp(min=1e-30)

    share = apply_alignment_weight_smoothing(share, logits_len, target_lens,
                                             alpha, beta, eps=eps)

    out = (share.clamp(min=1e-30).log() + take(peak)
           + take(total).clamp(min=1e-30).log())
    return torch.where(valid & take(live), out, log_x)


def apply_diagonal_projected_smoothing(target: torch.Tensor,
                                       gamma: torch.Tensor,
                                       labels: torch.Tensor,
                                       logits_len: torch.Tensor,
                                       target_lens: torch.Tensor,
                                       blank: int,
                                       alpha: float,
                                       beta: float,
                                       eps: float = 1e-10,
                                       share: str = "ratio_label_only") -> torch.Tensor:
    """DPS: FAS in the anti-diagonal's CLASS space, projected back to nodes.

    Diagonal Projected Smoothing. Per-node FAS keeps failing on RNN-T for
    a structural reason: z_hat(t, u, .) has at most two non-zero entries
    (blank and y_{u+1}), so the active set is 2 and there is nothing to
    redistribute among. This mode moves the smoothing to a space where
    the active set is genuinely wider, then pushes the result back down.

    The axis is the anti-diagonal t + u = k, because that is the one whose
    occupancy is a probability distribution: every transition raises t + u
    by exactly one, so a path crosses each anti-diagonal exactly once and
    `sum_{t+u=k} gamma(t, u) = 1` (verified to 4e-6). Columns are NOT
    normalized -- `sum_u gamma(t, u)` is the expected number of nodes
    visited in frame t, measured 1.00 to 2.69. So k indexes the alignment
    path's k-th STEP, and the aggregate below is "the distribution of the
    class emitted at step k" -- the exact RNN-T analogue of CTC's
    per-frame class posterior.

    Four steps:

    1. Aggregate to the diagonal's class distribution,

           z_k(j) = sum_{t+u=k} gamma(t, u) z_hat(t, u, j),

       which sums to one over j for free, since sum_{t+u=k} gamma = 1 and
       each node's target is normalized. Blank collects from EVERY node on
       the diagonal; a label class c collects from the nodes whose
       y_{u+1} is c, and a transcript that repeats c has several of those.

    2. Run the ordinary class-space FAS on z_k, BLANK INCLUDED. Blank
       usually dominates z_k, so it decides the active set, the active
       mass and therefore how much every other class is moved; dropping it
       would change the size of the smoothing, not just its support.

    3. Project back, REAL TOKENS ONLY. Scale each node's label component
       by its class's ratio r_k(c) = z_tilde_k(c) / z_k(c); blank is then
       fixed by 1 - (label), because a node has exactly those two
       non-zero entries. Multiplying the label component by a per-class
       ratio is what makes the nodes that can produce c share the change
       in proportion to gamma, which is the intended rule.

    4. No renormalization step: the construction in 3 already leaves every
       node summing to one. Blank's diagonal aggregate lands on
       z_tilde_k(blank) by itself --

           sum_j sum_{D_k} gamma z_hat' = sum_{D_k} gamma = 1
           sum_{D_k} gamma z_hat'(c)    = z_tilde_k(c)      (real c, by 3)
           => sum_{D_k} gamma z_hat'(blank) = 1 - sum_c z_tilde_k(c)
                                            = z_tilde_k(blank)

       since FAS preserves total mass. That identity is why blank needs no
       projection of its own. It holds up to the clamp in step 3, which
       fires when r_k(c) > 1 pushes an already near-one label component
       past one; that residual is accepted.

    Unlike `apply_alignment_weight_smoothing`, gamma itself is untouched,
    so this does not create AWS's self-referential loop (AWS training
    raised alignment entropy 3.46x and the active-node count 2.25 -> 5.01,
    and its RNN-T gain decayed away by the end of training). Only the
    target moves, which is the structure FAS wins with on CTC.

    Args:
        target: (B, T, U+1, C) per-node class target, normalized over C.
        gamma: (B, T, U+1) node occupancies, zero outside the rectangle.
        labels: (B, U) label ids, no blanks inserted.
        logits_len: (B,) valid frame count per sample.
        target_lens: (B,) real label count per sample.
        blank: blank class id.
        alpha: FAS rate in class space, in units of 1/C -- the same units
            as every other class-space mode here, NOT AWS's per-diagonal
            node units.
        beta: FAS floor height for classes the diagonal cannot reach.
        eps: a class is active when z_k exceeds it. MUST be > 0: at
            eps <= 0 every class is active, including the ones with no
            contributing node on the diagonal, and their alpha/C has no
            node to be projected onto -- that mass is simply lost. With
            eps > 0 an active class always has a contributing node.

    Returns:
        (B, T, U+1, C) smoothed target, normalized at every valid node and
        left exactly as passed in outside the rectangle.
    """
    assert eps > 0.0, (
        "diagonal_projected needs a strictly positive eps: at eps <= 0 the "
        f"classes with no node on the diagonal go active and lose their "
        f"mass in the projection. Got {eps}")
    b, t_len, u1, num_classes = target.shape
    device = target.device
    dtype = target.dtype
    n_diag = t_len + u1

    t_idx = torch.arange(t_len, device=device).view(-1, 1)
    u_idx = torch.arange(u1, device=device).view(1, -1)
    diag = (t_idx + u_idx).reshape(-1)                        # (T*(U+1),)

    valid = ((t_idx < logits_len.view(-1, 1, 1))
             & (u_idx <= target_lens.view(-1, 1, 1)))         # (B, T, U+1)
    # gamma is already exactly zero outside the rectangle; the mask is
    # belt-and-braces so a caller that hands in a smoothed gamma cannot
    # leak padding into the aggregate.
    w = gamma * valid.to(dtype)

    # --- 1. gamma-weighted aggregate over each anti-diagonal.
    z = torch.zeros(b, n_diag, num_classes, dtype=dtype, device=device)
    z.index_add_(1, diag, (w.unsqueeze(3) * target).reshape(b, -1,
                                                           num_classes))

    # --- 2. ordinary class-space FAS, blank included.
    full = torch.full((b,), n_diag, dtype=torch.long, device=device)
    z_smoothed = shc_loss_util.apply_floored_active_support_smoothing(
        z, full, alpha, beta, eps=eps)

    # --- 3. split each class's SMOOTHED MASS over the nodes that can
    # emit it, in proportion to their occupancy.
    #
    #     M(t, l, j) = z_tilde_k(j) * gamma(t, l) / G_k(j),
    #     G_k(j)     = sum over the diagonal's nodes with y_{l+1} = j
    #
    # and the node's CONDITIONAL is that mass divided by its own gamma,
    # so gamma cancels:
    #
    #     z_hat'(t, l, j) = z_tilde_k(j) / G_k(j)
    #
    # Every node that can emit j on this diagonal therefore gets the same
    # value. That is not a loss of the lattice's information in practice:
    # a diagonal almost always carries a given class at a single label
    # position, in which case G_k(j) = gamma(t, l) and the expression
    # reduces to z_hat*(1 - mass) + (alpha/K)/gamma -- the node's own
    # shape, shifted. Uniformity only bites where the transcript repeats a
    # class inside one diagonal.
    #
    # Note what is NOT done here: dividing by z_k(j) = sum gamma*z_hat.
    # That denominator carries the emission factor as well, so at a
    # blank-dominant node (z_hat ~ 1e-9) the ratio explodes and the
    # projection saturates the target at 1. Measured: WER went the wrong
    # way from step 1000 to step 2000 at both alpha = 0.05 and 0.10.
    # Dividing by the pure occupancy sum is bounded by comparison --
    # z_hat' <= 1 needs roughly G_k(j) >= 1/N_a.
    # Column u emits labels[:, u]; the last column has no label left. Its
    # id is set to blank there so the gathers below stay in range -- that
    # column is excluded by `label_col`.
    lab = labels.clamp(min=0)
    lab_full = torch.cat(
        [lab, torch.full((b, 1), blank, dtype=lab.dtype, device=device)],
        dim=1)                                                # (B, U+1)
    lab_node = lab_full.view(b, 1, u1).expand(b, t_len, u1)   # (B, T, U+1)
    label_col = valid & (u_idx < target_lens.view(-1, 1, 1))

    # G_k(j): occupancy summed over the diagonal's nodes that emit j.
    # Scattered by (diagonal, class) in one flat index_add_ rather than
    # materializing a second (B, T, U+1, C) tensor.
    lin = (diag.view(1, -1) * num_classes
           + lab_node.reshape(b, -1))                         # (B, T*(U+1))
    gsum = torch.zeros(b, n_diag * num_classes, dtype=dtype, device=device)
    gsum.scatter_add_(1, lin, (w * label_col.to(dtype)).reshape(b, -1))

    # Only the INCREMENT is shared out, not the class's whole mass:
    #
    #     z_hat'(t, l, j) = z_hat(t, l, j) + delta_k(j) / G_k(j),
    #     delta_k(j)      = z_tilde_k(j) - z_k(j)
    #
    # The mass node (t, l) receives is gamma(t, l) * delta_k(j) / G_k(j),
    # i.e. proportional to its occupancy, which is the intended rule; and
    # the aggregate still lands exactly on the smoothed value, since
    # sum gamma * z_hat' = z_k(j) + delta_k(j) * (sum gamma / G_k) =
    # z_tilde_k(j).
    #
    # Sharing out the class's TOTAL mass instead gives
    # z_hat' = z_tilde_k(j) / G_k(j), which is the gamma-weighted MEAN of
    # z_hat over the nodes that emit j -- so at alpha = 0 it replaces each
    # node's own value by that mean rather than leaving it alone. The
    # distortion does not scale with alpha and does not vanish with it:
    # measured 0.0012 in gamma-weighted L1 on a trained 1hr model at
    # alpha = 0, which is most of the 0.0014 gap to node-level FAS at
    # alpha = 0.01. Splitting the increment removes it exactly, and is the
    # only form that is both gamma-proportional and an identity at
    # alpha = 0.
    if share == "ratio":
        # Multiply every class by its own coefficient and renormalize the
        # node. w_k(j) = z_tilde_k(j) / z_k(j) is the factor by which FAS
        # moved class j on this diagonal; blank has one too and gets it.
        # Nothing is clipped, so nothing is thrown away -- the node stays
        # a distribution by construction.
        ratio = torch.where(z > eps, z_smoothed / z.clamp(min=eps),
                            torch.ones_like(z)).reshape(b, -1)
        w_lab = ratio.gather(1, lin).reshape(b, t_len, u1)
        w_bl = ratio.gather(
            1, diag.view(1, -1).expand(b, -1) * num_classes + blank
        ).reshape(b, t_len, u1)
        p_lab = torch.gather(target, 3, lab_node.unsqueeze(3)).squeeze(3)
        p_bl = target[..., blank]
        a_lab = p_lab * w_lab
        a_bl = p_bl * w_bl
        denom = (a_lab + a_bl).clamp(min=1e-30)
        p_new = torch.where(label_col, a_lab / denom, p_lab)
        out = torch.zeros_like(target)
        out[..., blank] = 1.0 - p_new
        out.scatter_add_(3, lab_node.unsqueeze(3), p_new.unsqueeze(3))
        return torch.where(valid.unsqueeze(3), out, target)

    if share == "ratio_label_only":
        # Solve one coefficient w_j per (diagonal, class) and scale every
        # node that emits j by it:  z_hat' = w_j * z_hat, with w_j fixed
        # by the aggregate, sum gamma * w_j * z_hat = z_tilde_k(j), i.e.
        # w_j = z_tilde_k(j) / z_k(j) -- exactly the factor by which FAS
        # moved the class.
        w_j = (z_smoothed / z.clamp(min=eps)).reshape(b, -1).gather(1, lin)
        p_lab = torch.gather(target, 3, lab_node.unsqueeze(3)).squeeze(3)
        p_new = (p_lab.reshape(b, -1) * w_j).reshape(b, t_len, u1)
        ok = label_col & (z.reshape(b, -1).gather(1, lin).reshape(
            b, t_len, u1) > eps)
    else:
        delta = (z_smoothed - z).reshape(b, -1).gather(1, lin)
        gnew = gsum.gather(1, lin)
        p_lab = torch.gather(target, 3, lab_node.unsqueeze(3)).squeeze(3)
        p_new = (p_lab.reshape(b, -1) + delta / gnew.clamp(min=eps)
                 ).reshape(b, t_len, u1)
        ok = label_col & (gnew.reshape(b, t_len, u1) > eps)
    p_new = torch.where(ok, p_new.clamp(0.0, 1.0), p_lab)

    # --- 4. rebuild; blank takes whatever the label component left.
    out = torch.zeros_like(target)
    out[..., blank] = 1.0 - p_new
    out.scatter_add_(3, lab_node.unsqueeze(3), p_new.unsqueeze(3))
    return torch.where(valid.unsqueeze(3), out, target)


def uniform_acoustic_node_target(logits_len: torch.Tensor,
                                 target_lens: torch.Tensor,
                                 labels: torch.Tensor,
                                 t_len: int,
                                 u1: int,
                                 num_classes: int,
                                 blank: int,
                                 dtype: torch.dtype) -> torch.Tensor:
    """The node target the lattice would give if the acoustics said nothing.

    With every transition equally likely, all RNN-T paths have the same
    length T + U and hence the same probability, so the alignment posterior
    is the ratio of path counts:

        gamma(t,u) = C(t+u, u) C(T-1-t+U-u, U-u) / C(T-1+U, U)

    verified against the recursion to 1e-15. What the smoothing needs is
    the NODE-CONDITIONAL, and that ratio collapses to something with no
    binomials left in it:

        u(blank | t,u)   = (T-1-t) / ((T-1-t) + (U-u))
        u(y_{u+1} | t,u) = (U-u)   / ((T-1-t) + (U-u))

    i.e. "remaining frames : remaining labels". That is the maximum-entropy
    alignment -- spend the budget evenly -- and it is the reference
    `alignment_biased` pulls the target toward.

    Note its support is exactly {blank, y_{u+1}}, the same two classes the
    node's own target has. So unlike every attempt to widen the active set
    (per-node FAS, per-frame, per-anti-diagonal), this puts ZERO mass on a
    label that is wrong given the node's predictor history: it re-weights
    the blank/label split and nothing else. It regularizes WHERE the labels
    are emitted, not WHICH labels they are.

    At the terminal node the two remainders are both zero; the only legal
    action there is the exit blank, so the reference is all blank.

    Args:
        logits_len: (B,) valid frame count per sample.
        target_lens: (B,) real label count per sample.
        labels: (B, U) label ids without blanks.
        t_len: padded T.
        u1: padded U + 1.
        num_classes: C.
        blank: blank class id.
        dtype: dtype of the returned tensor.

    Returns:
        (B, T, U+1, C) reference, normalized over classes at every node.
    """
    device = logits_len.device
    b = logits_len.shape[0]
    t_ar = torch.arange(t_len, device=device).view(1, -1, 1)
    u_ar = torch.arange(u1, device=device).view(1, 1, -1)
    rem_f = (logits_len.view(-1, 1, 1) - 1 - t_ar).clamp(min=0).to(dtype)
    rem_l = (target_lens.view(-1, 1, 1) - u_ar).clamp(min=0).to(dtype)
    total = rem_f + rem_l
    # total == 0 only at the terminal node, where the exit blank is the one
    # legal action.
    blank_share = torch.where(total > 0, rem_f / total.clamp(min=1e-30),
                              torch.ones_like(total))
    ref = torch.zeros((b, t_len, u1, num_classes), device=device, dtype=dtype)
    ref[..., blank] = blank_share
    lab = labels.clamp(min=0)
    u_real = lab.shape[1]
    lab_idx = lab.view(b, 1, u_real, 1).expand(b, t_len, u_real, 1)
    ref[:, :, :u_real, :].scatter_add_(
        3, lab_idx, (1.0 - blank_share[:, :, :u_real]).unsqueeze(3))
    return ref


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
                diag_align="departure",
                gate="none",
                gate_thresh=0.9,
                sharpen=0.0,
                sharpen_thresh=0.9):
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
            sharpen: target sharpening strength, the opposite of
                smoothing. At every node whose target already puts more
                than `sharpen_thresh` on one class, the target is moved
                that fraction of the way to a one-hot on that class; 1.0
                snaps it all the way. Because the node target is the
                alignment posterior, this interpolates the loss from
                marginalization (Baum-Welch) toward hard alignment
                (Viterbi), which is the direction smoothing moves away
                from. It self-limits early in training, when no node's
                target is concentrated enough to qualify. Applies to
                blank-dominant and label-dominant nodes alike -- the
                argmax decides which -- and is independent of `alpha`.
            sharpen_thresh: the confidence a node must already have
                before it is sharpened.
            gate: "none", "low", "high", "blank_only" or "label_only".
                "low"/"high" restrict smoothing to nodes whose UNSMOOTHED
                target max is at most (`low`) or above (`high`)
                `gate_thresh`.

                "blank_only"/"label_only" split on WHICH class dominates
                rather than by how much, and `gate_thresh` is ignored. The
                motivation is that FAS moves the blank target by

                    d target(blank) = (alpha / K) (1 - 2 z_blank)

                so it RAISES blank exactly at the nodes where a label has to
                be emitted, and lowers it where blank already dominates. In
                RNN-T each label is emitted once, so the first half is the
                deletion mechanism -- measured on 100 h at alpha = 0.05,
                deletions +18 %, insertions -28 %, substitutions flat.
                "blank_only" keeps just the second half, which both spares
                the emission decision and biases the remaining nodes toward
                emitting. Note this is no longer a symmetric regularizer but
                a directional one.

                The motivation is that smoothing here has a fixed point far
                below the one-step perturbation: the target is computed FROM
                the model, so a flatter model gives a flatter alignment
                posterior which gives a flatter target, and the loop runs for
                the whole schedule. Measured on 100 h at alpha = 0.05, the
                dominant class at a label node falls from 0.963 (baseline) to
                0.791 and its logit margin from 8.31 to 2.10 -- far past the
                0.9984 ceiling a fixed target would impose. "low" breaks that
                loop by exempting nodes the model is already sure about;
                "high" does the opposite and smooths only those, which is the
                reading that matches "penalize overconfidence".

                Note "low" is a one-way ratchet: a node that crosses the
                threshold stops being smoothed and is never pulled back, so
                it behaves like an alpha schedule that switches off as the
                model sharpens. On a converged baseline 94.5 % of the node
                occupancy already sits above 0.9.
            gate_thresh: the threshold itself.
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
        assert gate in ("none", "low", "high", "blank_only",
                        "label_only"), gate
        assert alpha_mode in ("fixed", "active_support",
                              "floored_active_support",
                              "frame_label_support",
                              "diagonal_active_support",
                              # "diagonal_occupancy" is the old name for
                              # "aws", kept so a chain launched before the
                              # rename does not die mid-sweep.
                              "aws", "diagonal_occupancy", "mos",
                              "diagonal_projected", "aws_alpha_beta",
                              "alignment_biased", "asap"), alpha_mode

        b, t_len, u1, num_classes = logits.shape
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        if alpha > 0.0 and alpha_mode == "mos":
            # MOS -- see shc_loss.py. The joint network's output is
            # smoothed first and alpha/beta/gamma/z_hat are all derived
            # from the smoothed distribution, so unlike every other mode
            # here this one does not touch the target directly.
            bsz, t_n, u_n, c_n = logits.shape
            flat = log_probs.exp().reshape(bsz, t_n * u_n, c_n)
            # Validity along the flattened node axis is a (T_b, U_b)
            # rectangle, not a prefix, so the full width is passed here and
            # gamma -- exactly zero outside the rectangle -- does the real
            # masking, exactly as the per-node smoothing modes below do.
            full = torch.full((bsz,), t_n * u_n, dtype=torch.long,
                              device=logits.device)
            flat = shc_loss_util.apply_floored_active_support_smoothing(
                flat, full, alpha, beta, eps=fas_eps)
            log_probs = flat.reshape(logits.shape).clamp(min=1e-30).log()

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

        if alpha > 0.0 and alpha_mode == "aws_alpha_beta":
            # AWS on BOTH lattice halves, before anything is derived from
            # them. alpha cancels out of z_hat, so smoothing it moves only
            # the per-node weight; beta enters z_hat, so smoothing it moves
            # the weight and the target together. The loss is left as the
            # unsmoothed lattice computed it -- this is a gradient-side
            # intervention, exactly as plain AWS is.
            log_alpha = apply_alpha_beta_diagonal_smoothing(
                log_alpha, logits_len, target_lens, alpha, beta, eps=fas_eps)
            log_beta = apply_alpha_beta_diagonal_smoothing(
                log_beta, logits_len, target_lens, alpha, beta, eps=fas_eps)

        q_blank, q_label = rnnt_transition_posteriors(
            log_alpha, log_beta, log_p_blank, log_p_label, log_seq_prob,
            logits_len, target_lens)
        target, gamma = _node_target(q_blank, q_label, lab, num_classes,
                                     blank)

        if alpha > 0.0 and alpha_mode == "aws_alpha_beta":
            # gamma = alpha*beta/P is a probability over each anti-diagonal
            # -- a path crosses every diagonal exactly once, so the sum is
            # one (measured 0.9996 to 1.0004 on a real batch). Smoothing
            # the two halves SEPARATELY does not preserve that: each half
            # keeps its own diagonal mass, but their product does not. A
            # node whose share of a diagonal was 1e-30 is lifted to the
            # floor alpha/K ~ 3e-4, a factor of 1e26, and when that happens
            # in both halves at once the product runs away -- measured
            # diagonal sums of 6.8e6 on average and 2.0e8 at worst at
            # alpha = 0.01, with gamma reaching 3.3e7. Training went to NaN
            # by step 1250 at both 0.01 and 0.02, WER 1.0.
            #
            # Restoring the invariant costs nothing that the mode wants:
            # z_hat is q_blank/(q_blank+q_label) WITHIN a node, so a
            # per-diagonal rescale of gamma leaves the target -- the part
            # beta actually moves -- exactly as it was.
            t_ar = torch.arange(t_len, device=logits.device).view(-1, 1)
            u_ar = torch.arange(u1, device=logits.device).view(1, -1)
            d_ix = (t_ar + u_ar).reshape(-1)
            d_sum = torch.zeros(b, t_len + u1, dtype=gamma.dtype,
                                device=logits.device)
            d_sum.index_add_(1, d_ix, gamma.reshape(b, -1))
            gamma = gamma / d_sum.clamp(min=1e-30).index_select(
                1, d_ix).reshape(b, t_len, u1)

        smoothing_enabled = alpha > 0.0
        unsmoothed = target if gate != "none" else None
        if smoothing_enabled and alpha_mode == "alignment_biased":
            reference = uniform_acoustic_node_target(
                logits_len, target_lens, lab, t_len, u1, num_classes,
                blank, target.dtype)
            target = (1.0 - alpha) * target + alpha * reference
        elif smoothing_enabled and alpha_mode == "diagonal_active_support":
            target = apply_diagonal_active_support_smoothing(
                target, gamma, alpha, beta, eps=fas_eps, align=diag_align)
        elif smoothing_enabled and alpha_mode == "diagonal_projected":
            # Needs (t, u) structure, the occupancy and the label ids, so
            # it does not go through the flattened per-node path below.
            target = apply_diagonal_projected_smoothing(
                target, gamma, lab, logits_len, target_lens, blank,
                alpha, beta, eps=fas_eps)
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

        if smoothing_enabled and gate != "none":
            if gate in ("blank_only", "label_only"):
                blank_dominant = (unsmoothed[..., blank] > 0.5).unsqueeze(3)
                keep = ~blank_dominant if gate == "blank_only" else blank_dominant
            else:
                mx = unsmoothed.max(dim=3, keepdim=True).values
                keep = ((mx > gate_thresh) if gate == "low"
                        else (mx <= gate_thresh))
            target = torch.where(keep, unsmoothed, target)

        if sharpen > 0.0:
            # Nodes outside a sample's (T_b, U_b) rectangle hold an all-zero
            # target, so their max is 0 and they never qualify; gamma is
            # zero there in any case.
            mx, idx = target.max(dim=3, keepdim=True)
            one_hot = torch.zeros_like(target).scatter_(3, idx, 1.0)
            hot = (1.0 - sharpen) * target + sharpen * one_hot
            target = torch.where(mx > sharpen_thresh, hot, target)

        if smoothing_enabled and alpha_mode in ("aws", "diagonal_occupancy"):
            # The one mode that leaves `target` alone: it reweights nodes
            # instead of rewriting what they are trained toward.
            gamma = apply_alignment_weight_smoothing(
                gamma, logits_len, target_lens, alpha, beta, eps=fas_eps)

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
        # fas_eps, asap_eps, diag_align, gate, gate_thresh, sharpen,
        # sharpen_thresh. Only `logits` (position 3) gets a gradient.
        return (None, None, gradient, None, None, None, None, None, None,
                None, None, None, None, None, None)


def rnnt_shc_loss(labels, target_lens, logits, logits_len, blank=0,
                  alpha=0.0, beta=0.0, alpha_mode="fixed",
                  fas_eps=1e-10, asap_eps=1e-3, reduction="mean",
                  diag_align="departure", gate="none", gate_thresh=0.9,
                  sharpen=0.0, sharpen_thresh=0.9):
    """Thin functional wrapper, mirroring torchaudio.functional.rnnt_loss."""
    loss = RnntShcLoss.apply(labels, target_lens, logits, logits_len, blank,
                             alpha, beta, alpha_mode, fas_eps, asap_eps,
                             diag_align, gate, gate_thresh, sharpen,
                             sharpen_thresh)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction {reduction!r}")
