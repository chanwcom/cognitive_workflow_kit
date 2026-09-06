"""Tests for the "hybrid" smoothing space of ShcLoss.

"hybrid" mixes the target toward

    (1 - beta) * uniform over the C classes
  + beta       * the label-space masked uniform, scattered into class space

so its two endpoints are each something a reader already has a name for:

    beta = 0   textbook uniform label smoothing, (1 - alpha) * y + alpha / C
    beta = 1   the historical L-SETS prior, blank-heavy and
               occurrence-weighted, byte-for-byte the "label" space result

Only the beta = 0 end moves relative to "label" space, which is the point:
there both ends of the blend were built on the label axis, so beta = 0 was
a uniform over label POSITIONS -- roughly half of them blank -- rather than
a uniform over classes. Both properties are asserted directly below, since
the whole reason to run this grid is that beta now interpolates between two
known methods rather than between one known and one accidental one.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import unittest

# Third-party imports
import torch

# Custom imports
from cwk.loss.pytorch import seq_loss_util, shc_loss, shc_loss_util


def _fixture(seed=0, batch=3, max_time=40, num_classes=12, target_len=6):
    """Builds a small but structurally valid (logits, labels) fixture."""
    torch.manual_seed(seed)
    logits = torch.randn(batch, max_time, num_classes, requires_grad=True)
    # Label ids start at 1: 0 is the blank index used by ShcLoss.
    labels = torch.randint(1, num_classes, (batch, target_len))
    target_lens = torch.full((batch,), target_len, dtype=torch.int32)
    logits_len = torch.tensor([max_time, max_time - 5, max_time - 10],
                              dtype=torch.int32)[:batch]
    return logits, labels, target_lens, logits_len


def _grad(logits, labels, target_lens, logits_len, space, alpha, beta):
    """Runs one forward/backward through ShcLoss and returns the gradient."""
    if logits.grad is not None:
        logits.grad = None
    loss = shc_loss.ShcLoss.apply(
        labels, target_lens, logits.log_softmax(2), logits_len,
        logits.shape[2], alpha, beta, False, 0.0, False, space).mean()
    loss.backward()
    return logits.grad.clone()


def _gamma_and_clamped(logits, labels, target_lens, logits_len):
    """Recomputes the alignment posterior, independently of ShcLoss.forward."""
    augmented = seq_loss_util.to_blank_augmented_labels(
        {"SEQ_DATA": labels, "SEQ_LEN": target_lens}, 0, False, False)
    clamped = torch.clamp(augmented["SEQ_DATA"], min=0)
    log_probs = logits.log_softmax(2).detach()
    trans = seq_loss_util.label_trans_allowance_table_ctc(
        augmented["SEQ_DATA"], augmented["SEQ_LEN"])
    log_label_probs = seq_loss_util.calculate_log_label_prob(
        clamped, log_probs)
    log_alpha, log_beta, _ = shc_loss.calculate_alpha_beta(
        trans, log_label_probs, augmented["SEQ_LEN"], logits_len)
    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - torch.logsumexp(log_gamma, axis=2, keepdim=True)
    return torch.exp(log_gamma), clamped, log_probs


def _hybrid_target(gamma, clamped, log_probs, logits_len, alpha, beta):
    """Reference: (1-beta) * class uniform + beta * scattered masked uniform."""
    target = shc_loss._scatter_to_class_space(gamma, log_probs, clamped)
    active = (gamma >= 1e-6).to(gamma.dtype)
    mix_label = active / active.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mix_prob = shc_loss._scatter_to_class_space(mix_label, log_probs, clamped)
    return shc_loss_util.apply_mixed_prior_smoothing(
        target, mix_prob, logits_len, alpha, beta)


def _label_target(gamma, clamped, log_probs, logits_len, alpha, beta):
    """Reference: smooth on the label axis, then scatter."""
    smoothed = shc_loss_util.apply_post_processing(
        gamma, logits_len, alpha, beta)
    return shc_loss._scatter_to_class_space(smoothed, log_probs, clamped)


class HybridSpaceTest(unittest.TestCase):

    def test_beta_zero_is_textbook_uniform_label_smoothing(self):
        """beta = 0 must be exactly (1 - alpha) * y + alpha / C."""
        logits, labels, target_lens, logits_len = _fixture()
        gamma, clamped, log_probs = _gamma_and_clamped(
            logits, labels, target_lens, logits_len)
        num_classes = logits.shape[2]
        unsmoothed = shc_loss._scatter_to_class_space(
            gamma, log_probs, clamped)
        time_idx = torch.arange(logits.shape[1])
        valid = (time_idx.unsqueeze(0)
                 < logits_len.unsqueeze(1)).unsqueeze(-1)

        for alpha in (0.01, 0.05, 0.2):
            expected = ((1.0 - alpha) * unsmoothed
                        + alpha / num_classes) * valid
            actual = _hybrid_target(
                gamma, clamped, log_probs, logits_len, alpha, 0.0)
            self.assertLess((actual - expected).abs().max().item(), 1e-6)

    def test_beta_one_reproduces_label_space(self):
        """beta = 1 must keep the historical L-SETS prior exactly.

        Scatter is linear, so blending in class space after scattering the
        masked uniform equals blending on the label axis and scattering.
        """
        logits, labels, target_lens, logits_len = _fixture()
        gamma, clamped, log_probs = _gamma_and_clamped(
            logits, labels, target_lens, logits_len)
        for alpha in (0.01, 0.05, 0.2):
            hybrid = _hybrid_target(
                gamma, clamped, log_probs, logits_len, alpha, 1.0)
            label = _label_target(
                gamma, clamped, log_probs, logits_len, alpha, 1.0)
            self.assertLess((hybrid - label).abs().max().item(), 1e-6)

    def test_beta_one_matches_label_space_end_to_end(self):
        """The same equality, but through ShcLoss's own dispatch."""
        logits, labels, target_lens, logits_len = _fixture()
        hybrid = _grad(logits, labels, target_lens, logits_len,
                       "hybrid", 0.05, 1.0)
        label = _grad(logits, labels, target_lens, logits_len,
                      "label", 0.05, 1.0)
        self.assertLess((hybrid - label).abs().max().item(), 1e-6)

    def test_beta_zero_differs_from_label_space(self):
        """...and beta = 0 is exactly where the two must part company."""
        logits, labels, target_lens, logits_len = _fixture()
        hybrid = _grad(logits, labels, target_lens, logits_len,
                       "hybrid", 0.05, 0.0)
        label = _grad(logits, labels, target_lens, logits_len,
                      "label", 0.05, 0.0)
        self.assertGreater((hybrid - label).abs().max().item(), 1e-4)

    def test_blank_mass_of_the_two_endpoints(self):
        """Pins the numbers the write-up quotes for the prior itself.

        alpha = 1 erases y, so the smoothed target IS the mixing prior.
        """
        num_classes = 8
        # [3, blank, 5, blank, 3]: blank at 2 of 5 positions, class 3 at 2.
        clamped = torch.tensor([[3, 0, 5, 0, 3]], dtype=torch.long)
        gamma = torch.full((1, 1, 5), 0.2)
        log_probs = torch.zeros(1, 1, num_classes)
        logits_len = torch.tensor([1])

        prior0 = _hybrid_target(
            gamma, clamped, log_probs, logits_len, 1.0, 0.0)
        prior1 = _hybrid_target(
            gamma, clamped, log_probs, logits_len, 1.0, 1.0)

        # beta = 0: genuinely uniform over classes.
        self.assertAlmostEqual(prior0[0, 0, 0].item(), 1.0 / num_classes, 6)
        self.assertAlmostEqual(prior0[0, 0, 7].item(), 1.0 / num_classes, 6)
        # beta = 1: blank keeps a share per position (2/5), class 3 keeps
        # two shares because it occurs twice (2/5), class 5 one (1/5), and
        # classes absent from the transcript keep nothing.
        self.assertAlmostEqual(prior1[0, 0, 0].item(), 2.0 / 5.0, 6)
        self.assertAlmostEqual(prior1[0, 0, 3].item(), 2.0 / 5.0, 6)
        self.assertAlmostEqual(prior1[0, 0, 5].item(), 1.0 / 5.0, 6)
        self.assertAlmostEqual(prior1[0, 0, 7].item(), 0.0, 6)

    def test_target_is_a_distribution(self):
        """Rows must still sum to 1 for every beta."""
        logits, labels, target_lens, logits_len = _fixture()
        gamma, clamped, log_probs = _gamma_and_clamped(
            logits, labels, target_lens, logits_len)
        time_idx = torch.arange(logits.shape[1])
        valid = time_idx.unsqueeze(0) < logits_len.unsqueeze(1)
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            target = _hybrid_target(
                gamma, clamped, log_probs, logits_len, 0.05, beta)
            row_sums = target.sum(-1)
            self.assertLess(
                (row_sums[valid] - 1.0).abs().max().item(), 1e-5)

    def test_alpha_zero_is_a_no_op(self):
        """No smoothing must leave the historical gradient untouched."""
        logits, labels, target_lens, logits_len = _fixture()
        for beta in (0.0, 0.5, 1.0):
            hybrid = _grad(logits, labels, target_lens, logits_len,
                           "hybrid", 0.0, beta)
            label = _grad(logits, labels, target_lens, logits_len,
                          "label", 0.0, beta)
            self.assertLess((hybrid - label).abs().max().item(), 1e-6)

    def test_is_padding_independent_at_every_beta(self):
        """Widening the label axis must not move the smoothed target.

        "label" space with beta < 1 fails this: its beta = 0 end is 1/L
        over ALL label positions, and padding adds positions, so the same
        utterance gets a different target depending on which others shared
        its batch. Here the beta = 0 end is 1/C, which has no L in it, and
        the beta = 1 end is thresholded on gamma, which is 0 on padding.
        """
        logits, labels, target_lens, logits_len = _fixture()
        gamma, clamped, log_probs = _gamma_and_clamped(
            logits, labels, target_lens, logits_len)

        for beta in (0.0, 0.5, 1.0):
            base = _hybrid_target(
                gamma, clamped, log_probs, logits_len, 0.05, beta)
            for pad_len in (4, 25):
                wide_gamma = torch.cat(
                    [gamma, torch.zeros(gamma.shape[0], gamma.shape[1],
                                        pad_len, dtype=gamma.dtype)], dim=-1)
                wide_clamped = torch.cat(
                    [clamped, torch.zeros(clamped.shape[0], pad_len,
                                          dtype=clamped.dtype)], dim=-1)
                wide = _hybrid_target(wide_gamma, wide_clamped, log_probs,
                                      logits_len, 0.05, beta)
                self.assertLess((wide - base).abs().max().item(), 1e-6)

    def test_label_space_beta_zero_is_padding_dependent(self):
        """The contrast the test above exists to document."""
        logits, labels, target_lens, logits_len = _fixture()
        gamma, clamped, log_probs = _gamma_and_clamped(
            logits, labels, target_lens, logits_len)
        base = _label_target(
            gamma, clamped, log_probs, logits_len, 0.05, 0.0)
        wide_gamma = torch.cat(
            [gamma, torch.zeros(gamma.shape[0], gamma.shape[1], 25,
                                dtype=gamma.dtype)], dim=-1)
        wide_clamped = torch.cat(
            [clamped, torch.zeros(clamped.shape[0], 25,
                                  dtype=clamped.dtype)], dim=-1)
        wide = _label_target(
            wide_gamma, wide_clamped, log_probs, logits_len, 0.05, 0.0)
        self.assertGreater((wide - base).abs().max().item(), 1e-5)

    def test_gamma_is_exactly_zero_on_padded_label_positions(self):
        """The assumption everything else here rests on.

        Padded label positions carry id 0, which is blank -- so if the
        forward-backward left any mass there, the beta = 1 prior would
        leak it straight onto blank, the class most at risk of being
        over-weighted already. It does not: the mass is exactly 0.0, not
        merely small, so the >= eps test excludes those positions no
        matter how eps is chosen.
        """
        labels = torch.zeros(2, 9, dtype=torch.long)
        labels[0, :3] = torch.tensor([4, 7, 4])
        labels[1] = torch.randint(1, 12, (9,))
        target_lens = torch.tensor([3, 9], dtype=torch.int32)
        logits = torch.randn(2, 30, 12, requires_grad=True)
        logits_len = torch.tensor([30, 30], dtype=torch.int32)

        gamma, _, _ = _gamma_and_clamped(
            logits, labels, target_lens, logits_len)
        augmented = seq_loss_util.to_blank_augmented_labels(
            {"SEQ_DATA": labels, "SEQ_LEN": target_lens}, 0, False, False)
        valid_len = int(augmented["SEQ_LEN"][0])

        padded = gamma[0, :, valid_len:]
        self.assertGreater(padded.numel(), 0)
        self.assertTrue(bool((padded == 0.0).all()))

    def test_padding_does_not_change_the_gradient_end_to_end(self):
        """Same utterance alone vs. batched with a longer one.

        This is the strongest form of the padding claim: it goes through
        ShcLoss itself rather than a reconstruction, so a leak anywhere in
        the chain -- gamma, the active mask, the scatter of the mixing
        prior, the blend -- would show up. "label" space fails this at
        every beta < 1, by an amount proportional to (1 - beta), since the
        component that spreads over all L positions is the one weighted by
        (1 - beta).
        """
        num_classes, max_time = 12, 30
        short = torch.tensor([[4, 7, 4]])
        long_labels = torch.randint(1, num_classes, (9,))
        logits_short = torch.randn(1, max_time, num_classes)
        other = torch.randn(1, max_time, num_classes)

        def grad_alone(space, beta):
            lg = logits_short.clone().requires_grad_(True)
            shc_loss.ShcLoss.apply(
                short, torch.tensor([3], dtype=torch.int32),
                lg.log_softmax(2), torch.tensor([max_time],
                                                dtype=torch.int32),
                num_classes, 0.05, beta, False, 0.0, False,
                space).mean().backward()
            return lg.grad.clone()

        def grad_padded(space, beta):
            lg = torch.cat([logits_short, other], 0).requires_grad_(True)
            labels = torch.zeros(2, 9, dtype=torch.long)
            labels[0, :3] = short[0]
            labels[1] = long_labels
            shc_loss.ShcLoss.apply(
                labels, torch.tensor([3, 9], dtype=torch.int32),
                lg.log_softmax(2),
                torch.tensor([max_time, max_time], dtype=torch.int32),
                num_classes, 0.05, beta, False, 0.0, False,
                space).mean().backward()
            # .mean() over two samples halves each one's share.
            return lg.grad[0:1].clone() * 2.0

        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertLess(
                (grad_alone("hybrid", beta)
                 - grad_padded("hybrid", beta)).abs().max().item(), 1e-6,
                msg=f"hybrid leaked padding at beta={beta}")
        # Sensitivity: the same probe does catch the known "label" leak.
        self.assertGreater(
            (grad_alone("label", 0.0)
             - grad_padded("label", 0.0)).abs().max().item(), 1e-4)

    def test_rejects_entropy_matched_alpha_mode(self):
        """Entropy matching is only defined for the class-space solve."""
        logits, labels, target_lens, logits_len = _fixture()
        with self.assertRaises(AssertionError):
            shc_loss.ShcLoss.apply(
                labels, target_lens, logits.log_softmax(2), logits_len,
                logits.shape[2], 0.05, 1.0, False, 0.0, False, "hybrid",
                "entropy_matched")


if __name__ == "__main__":
    unittest.main()
