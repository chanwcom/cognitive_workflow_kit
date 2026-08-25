"""Unit tests for ShcLoss's `smoothing_space` argument.

`smoothing_space` selects WHICH AXIS the SETS-family post-processing is
applied to (see `shc_loss_util`'s module docstring):

    "label" -- smooth gamma (B, T, L) over blank-augmented label POSITIONS,
               then scatter L -> C. The historical default; every SETS /
               PP-SETS / PC-SETS result produced so far used this.
    "class" -- scatter L -> C first, then smooth (B, T, C) over actual
               output CLASSES.

The tests below pin down three things:
  1. The two spaces coincide exactly when no smoothing is requested
     (alpha = 0 and no peak variant), since the smoothing call is then
     skipped entirely and both paths reduce to scatter -> gradient.
  2. They genuinely DIFFER once smoothing is on -- they are different
     algorithms, not two spellings of one.
  3. The historical "label" path still matches an explicit, inline
     reference implementation of the original pre-refactor chain, so
     splitting `_compute_gradient` into two compiled halves did not
     perturb it.
"""

# pylint: disable=import-error, no-member, no-name-in-module

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


def _loss_and_grad(logits, labels, target_lens, logits_len, space, alpha,
                   beta):
    """Runs one forward/backward through ShcLoss and returns (loss, grad)."""
    if logits.grad is not None:
        logits.grad = None
    loss = shc_loss.ShcLoss.apply(
        labels, target_lens, logits.log_softmax(2), logits_len,
        logits.shape[2], alpha, beta, False, 0.0, False, space).mean()
    loss.backward()
    return loss.item(), logits.grad.clone()


def _class_space_target(logits, labels, target_lens, logits_len, alpha, beta):
    """Reference: gamma -> scatter -> smooth, i.e. the "class" space target."""
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

    target = shc_loss._scatter_to_class_space(
        torch.exp(log_gamma), log_probs, clamped)
    return shc_loss_util.apply_post_processing(
        target, logits_len, alpha, beta)


def _label_space_target(logits, labels, target_lens, logits_len, alpha, beta):
    """Reference: gamma -> smooth -> scatter, i.e. the original chain.

    Written out inline (rather than calling the production helpers as a
    unit) precisely so it can serve as an independent check that the
    "label" path was not perturbed by the refactor.
    """
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

    gamma = shc_loss_util.apply_post_processing(
        torch.exp(log_gamma), logits_len, alpha, beta)
    target = torch.zeros_like(log_probs)
    target.scatter_add_(
        2, clamped.unsqueeze(1).expand(-1, log_probs.shape[1], -1), gamma)
    return target


def _valid_frame_mask(logits_len, max_time):
    return torch.arange(max_time)[None, :] < logits_len[:, None]


class SmoothingSpaceTest(unittest.TestCase):
    """Tests for the `smoothing_space` argument of ShcLoss.forward."""

    def test_no_smoothing_makes_both_spaces_identical(self):
        """alpha = 0 skips smoothing, so the space choice cannot matter."""
        logits, labels, target_lens, logits_len = _fixture()
        loss_label, grad_label = _loss_and_grad(
            logits, labels, target_lens, logits_len, "label", 0.0, 0.0)
        loss_class, grad_class = _loss_and_grad(
            logits, labels, target_lens, logits_len, "class", 0.0, 0.0)

        self.assertAlmostEqual(loss_label, loss_class, places=6)
        self.assertTrue(torch.allclose(grad_label, grad_class, atol=1e-6))

    def test_smoothing_makes_the_two_spaces_differ(self):
        """With smoothing on, the spaces are different algorithms."""
        logits, labels, target_lens, logits_len = _fixture()
        _, grad_label = _loss_and_grad(
            logits, labels, target_lens, logits_len, "label", 0.02, 1.0)
        _, grad_class = _loss_and_grad(
            logits, labels, target_lens, logits_len, "class", 0.02, 1.0)

        self.assertFalse(torch.allclose(grad_label, grad_class, atol=1e-5))

    def test_class_space_target_is_a_distribution(self):
        """The smoothed class-space target must sum to 1 on valid frames."""
        logits, labels, target_lens, logits_len = _fixture()
        target = _class_space_target(
            logits, labels, target_lens, logits_len, 0.02, 1.0)

        valid = _valid_frame_mask(logits_len, logits.shape[1])
        sums = target.sum(-1)[valid]
        self.assertTrue(
            torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
        self.assertTrue(torch.all(target >= 0.0))

    def test_class_space_target_is_zero_on_padded_frames(self):
        """Padding must stay zeroed, exactly as in the label-space path."""
        logits, labels, target_lens, logits_len = _fixture()
        target = _class_space_target(
            logits, labels, target_lens, logits_len, 0.02, 1.0)

        padded = ~_valid_frame_mask(logits_len, logits.shape[1])
        self.assertTrue(torch.all(target[padded] == 0.0))

    def test_label_space_path_matches_original_chain(self):
        """Splitting _compute_gradient must not perturb the "label" path.

        Compares the production path's gradient against one rebuilt from
        an inline gamma -> smooth -> scatter reference.
        """
        logits, labels, target_lens, logits_len = _fixture()
        _, grad_label = _loss_and_grad(
            logits, labels, target_lens, logits_len, "label", 0.02, 1.0)

        target = _label_space_target(
            logits, labels, target_lens, logits_len, 0.02, 1.0)
        log_probs = logits.log_softmax(2).detach()
        seq_mask = _valid_frame_mask(logits_len, logits.shape[1])
        expected = -(target - log_probs.exp())
        expected = expected * seq_mask.unsqueeze(2).to(expected.dtype)
        # `.mean()` over the batch in _loss_and_grad scales every sample's
        # gradient by 1 / batch_size.
        expected = expected / logits.shape[0]

        self.assertTrue(torch.allclose(grad_label, expected, atol=1e-5))

    def test_defaults_to_label_space(self):
        """Omitting the argument must reproduce the historical behavior."""
        logits, labels, target_lens, logits_len = _fixture()
        _, grad_explicit = _loss_and_grad(
            logits, labels, target_lens, logits_len, "label", 0.02, 1.0)

        logits.grad = None
        loss = shc_loss.ShcLoss.apply(
            labels, target_lens, logits.log_softmax(2), logits_len,
            logits.shape[2], 0.02, 1.0).mean()
        loss.backward()

        self.assertTrue(torch.allclose(grad_explicit, logits.grad, atol=1e-6))

    def test_rejects_unknown_space(self):
        """A typo'd space must fail loudly rather than silently fall back."""
        logits, labels, target_lens, logits_len = _fixture()
        with self.assertRaises(AssertionError):
            shc_loss.ShcLoss.apply(
                labels, target_lens, logits.log_softmax(2), logits_len,
                logits.shape[2], 0.02, 1.0, False, 0.0, False, "classes")

    def test_peak_variants_run_in_class_space(self):
        """PP-/PC-SETS must also route through the class-space path."""
        logits, labels, target_lens, logits_len = _fixture()
        for peak_preserving, peak_capping, alpha, gamma in (
                (True, False, 0.0, 0.1), (False, True, 0.05, 0.0)):
            logits.grad = None
            loss = shc_loss.ShcLoss.apply(
                labels, target_lens, logits.log_softmax(2), logits_len,
                logits.shape[2], alpha, 0.0, peak_preserving, gamma,
                peak_capping, "class").mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.all(torch.isfinite(logits.grad)))


class ScatterToClassSpaceTest(unittest.TestCase):
    """Tests for the L -> C scatter helper on its own."""

    def test_scatter_preserves_total_mass(self):
        """Summing label positions by class must not create/destroy mass."""
        torch.manual_seed(0)
        batch, max_time, num_classes, label_len = 2, 5, 7, 9
        gamma = torch.softmax(torch.randn(batch, max_time, label_len), dim=-1)
        log_probs = torch.zeros(batch, max_time, num_classes)
        labels = torch.randint(0, num_classes, (batch, label_len))

        out = shc_loss._scatter_to_class_space(gamma, log_probs, labels)

        self.assertEqual(out.shape, (batch, max_time, num_classes))
        self.assertTrue(torch.allclose(
            out.sum(-1), gamma.sum(-1), atol=1e-6))

    def test_repeated_labels_accumulate(self):
        """Every position carrying class c must add into that one class."""
        # All 4 label positions carry class 2, so all mass lands there.
        gamma = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
        log_probs = torch.zeros(1, 1, 5)
        labels = torch.tensor([[2, 2, 2, 2]])

        out = shc_loss._scatter_to_class_space(gamma, log_probs, labels)

        expected = torch.tensor([[[0.0, 0.0, 1.0, 0.0, 0.0]]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
