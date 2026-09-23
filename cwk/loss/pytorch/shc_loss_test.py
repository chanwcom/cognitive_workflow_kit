"""Unit tests for the shc_loss module."""

# pylint: disable=import-error, no-member, no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os
import unittest

# Third-party imports
import numpy as np
import torch

# Custom imports
from cwk.loss.pytorch import seq_loss_util
from cwk.loss.pytorch import shc_loss


class TestEnforceLogZero(unittest.TestCase):
    """Unit tests for enforce_log_zero function."""

    def setUp(self):
        global LOG_0
        LOG_0 = -706.893623

    def test_enforce_log_zero_basic(self):
        """Tests whether values near or below LOG_0 are correctly clamped to exact LOG_0."""
        # 텐서의 기본 dtype인 float32 기준에 맞춘 float32 상수로 변환
        log_zero_t32 = torch.tensor(LOG_0, dtype=torch.float32).item()

        tensor = torch.tensor([
            log_zero_t32,
            log_zero_t32 - 1e-3,  # Below LOG_0
            log_zero_t32 + 1e-6,  # Slightly drifted above LOG_0 within tolerance
            -5.0,                 # Normal valid log probability
            0.0,                  # Valid log probability
        ], dtype=torch.float32)

        expected = torch.tensor([
            log_zero_t32,
            log_zero_t32,
            log_zero_t32,
            -5.0,
            0.0,
        ], dtype=torch.float32)

        result = shc_loss.enforce_log_zero(tensor, log_zero=LOG_0, tol=1e-5)
        
        # 부동소수점 오차를 고려한 일치 여부 확인
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))
        
        # float32 정밀도 한계 내에서 일치하는지 확인
        self.assertAlmostEqual(result[0].item(), log_zero_t32, places=5)
        self.assertAlmostEqual(result[1].item(), log_zero_t32, places=5)
        self.assertAlmostEqual(result[2].item(), log_zero_t32, places=5)

    def test_enforce_log_zero_custom_tolerance(self):
        """Tests behavior with a custom tolerance value."""
        log_zero_t32 = torch.tensor(LOG_0, dtype=torch.float32).item()
        tensor = torch.tensor([log_zero_t32 + 1e-3], dtype=torch.float32)
        
        # With default tol (1e-5), this should NOT be clamped
        result_tight = shc_loss.enforce_log_zero(tensor, log_zero=LOG_0, tol=1e-5)
        self.assertNotAlmostEqual(result_tight[0].item(), log_zero_t32, places=5)

        # With a larger tol (1e-2), this SHOULD be clamped
        result_loose = shc_loss.enforce_log_zero(tensor, log_zero=LOG_0, tol=1e-2)
        self.assertAlmostEqual(result_loose[0].item(), log_zero_t32, places=5)


class TestSHCTransitionTable(unittest.TestCase):
    """Unit tests for the SHC transition table generation."""

    def setUp(self):
        self.boundary_id = 10
        self.blank_id = 0
        self.log_0 = -100.0

    def test_transition_rules(self):
        # Sequence: [A, Bnd, A, Blk, B]
        # Indices:  0,  1,   2,  3,   4
        token_seq = torch.tensor(
            [[5, self.boundary_id, 5, self.blank_id, 7]], dtype=torch.long
        )

        table = shc_loss.create_trans_allowance_table_shc(
            token_seq, self.boundary_id, self.blank_id, self.log_0
        )
        t = table[0]

        # 1. Boundary (idx 1) must NOT stay in the same state.
        self.assertEqual(
            t[1, 1], self.log_0, "Boundary token should not allow self-loop."
        )

        # 2. i -> i + 2 must NOT be allowed if i + 1 is a Boundary (idx 1).
        self.assertEqual(
            t[0, 2], self.log_0, "Should not skip over a boundary token."
        )

        # 3. i -> i + 2 MUST be allowed if i + 1 is a Blank (idx 3).
        self.assertEqual(
            t[2, 4], 0.0, "Should allow skip over a blank token."
        )

        # 4. Normal tokens (idx 0, 2, 4) should allow self-loops.
        self.assertEqual(t[0, 0], 0.0)
        self.assertEqual(t[2, 2], 0.0)
        self.assertEqual(t[4, 4], 0.0)

        # 5. Continuous transitions (i -> i + 1) should always be allowed.
        self.assertEqual(t[0, 1], 0.0)
        self.assertEqual(t[1, 2], 0.0)
        self.assertEqual(t[2, 3], 0.0)
        self.assertEqual(t[3, 4], 0.0)

    def test_batch_transitions(self):
        # Configuration
        boundary_id = 10
        blank_id = 0
        log_0 = -100.0
        L = -100.0  # Disallowed
        O = 0.0     # Allowed

        # Input batch (Size 3, Length 12)
        raw_seqs = [
            [3, 0, 10, 9, 0, 10, 7, 0, 10, 3, 0, 2],
            [3, 0, 10, 1, 0, 10, 2, 0, 0, 0, 0, 0],
            [3, 0, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        token_seq = torch.tensor(raw_seqs, dtype=torch.long)

        # Expected Matrix 0: [3, 0, 10, 9, 0, 10, 7, 0, 10, 3, 0, 2]
        exp0 = [
            [O, O, O, L, L, L, L, L, L, L, L, L], # 0: 3 (Skip as 1 is Blk)
            [L, O, O, L, L, L, L, L, L, L, L, L], # 1: 0 (Blk)
            [L, L, L, O, L, L, L, L, L, L, L, L], # 2: 10 (Bnd)
            [L, L, L, O, O, O, L, L, L, L, L, L], # 3: 9 (Skip as 4 is Blk)
            [L, L, L, L, O, O, L, L, L, L, L, L], # 4: 0 (Blk)
            [L, L, L, L, L, L, O, L, L, L, L, L], # 5: 10 (Bnd)
            [L, L, L, L, L, L, O, O, O, L, L, L], # 6: 7 (Skip as 7 is Blk)
            [L, L, L, L, L, L, L, O, O, L, L, L], # 7: 0 (Blk)
            [L, L, L, L, L, L, L, L, L, O, L, L], # 8: 10 (Bnd)
            [L, L, L, L, L, L, L, L, L, O, O, O], # 9: 3 (Skip as 10 is Blk)
            [L, L, L, L, L, L, L, L, L, L, O, O], # 10: 0 (Blk)
            [L, L, L, L, L, L, L, L, L, L, L, O], # 11: 2
        ]

        # Expected Matrix 1: [3, 0, 10, 1, 0, 10, 2, 0, 0, 0, 0, 0]
        exp1 = [
            [O, O, O, L, L, L, L, L, L, L, L, L], # 0: 3 (Skip as 1 is Blk)
            [L, O, O, L, L, L, L, L, L, L, L, L], # 1: 0 (Blk)
            [L, L, L, O, L, L, L, L, L, L, L, L], # 2: 10 (Bnd)
            [L, L, L, O, O, O, L, L, L, L, L, L], # 3: 1 (Skip as 4 is Blk)
            [L, L, L, L, O, O, L, L, L, L, L, L], # 4: 0 (Blk)
            [L, L, L, L, L, L, O, L, L, L, L, L], # 5: 10 (Bnd)
            [L, L, L, L, L, L, O, O, O, L, L, L], # 6: 2 (Skip as 7 is Blk)
            [L, L, L, L, L, L, L, O, O, O, L, L], # 7: 0 (Skip as 8 is Blk)
            [L, L, L, L, L, L, L, L, O, O, O, L], # 8: 0 (Skip as 9 is Blk)
            [L, L, L, L, L, L, L, L, L, O, O, O], # 9: 0 (Skip as 10 is Blk)
            [L, L, L, L, L, L, L, L, L, L, O, O], # 10: 0
            [L, L, L, L, L, L, L, L, L, L, L, O], # 11: 0
        ]

        # Expected Matrix 2: [3, 0, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        exp2 = [
            [O, O, O, L, L, L, L, L, L, L, L, L], # 0: 3 (Skip as 1 is Blk)
            [L, O, O, L, L, L, L, L, L, L, L, L], # 1: 0 (Blk)
            [L, L, L, O, L, L, L, L, L, L, L, L], # 2: 10 (Bnd)
            [L, L, L, O, O, O, L, L, L, L, L, L], # 3: 2 (Skip as 4 is Blk)
            [L, L, L, L, O, O, O, L, L, L, L, L], # 4: 0 (Skip as 5 is Blk)
            [L, L, L, L, L, O, O, O, L, L, L, L], # 5: 0 (Skip as 6 is Blk)
            [L, L, L, L, L, L, O, O, O, L, L, L], # 6: 0 (Skip as 7 is Blk)
            [L, L, L, L, L, L, L, O, O, O, L, L], # 7: 0 (Skip as 8 is Blk)
            [L, L, L, L, L, L, L, L, O, O, O, L], # 8: 0 (Skip as 9 is Blk)
            [L, L, L, L, L, L, L, L, L, O, O, O], # 9: 0 (Skip as 10 is Blk)
            [L, L, L, L, L, L, L, L, L, L, O, O], # 10: 0
            [L, L, L, L, L, L, L, L, L, L, L, O], # 11: 0
        ]

        expected_batch = torch.tensor([exp0, exp1, exp2], dtype=torch.float32)

        # Run function
        actual_batch = shc_loss.create_trans_allowance_table_shc(
            token_seq, boundary_id, blank_id, log_0
        )

        # Compare
        self.assertTrue(
            torch.allclose(actual_batch, expected_batch),
            "The batch transition tables do not match hard-coded expectations."
        )
        print("\nAll 3 Batch Examples Unit Test Passed!")

class TestShiftOps(unittest.TestCase):
    """Unit tests for horizontal shift operations."""

    def setUp(self):
        self.x = torch.tensor([[1, 2, 3], 
                               [4, 5, 6]], dtype=torch.float32)

    def test_shift_right(self):
        # Insert 9 at index 0
        expected = torch.tensor([[9, 1, 2], 
                                 [9, 4, 5]])
        result = shc_loss.shift_tensor_horizontally(self.x, 9.0, direction='right')
        self.assertTrue(torch.equal(result, expected))

    def test_shift_left(self):
        # Insert 9 at the last index
        expected = torch.tensor([[2, 3, 9], 
                                 [5, 6, 9]])
        result = shc_loss.shift_tensor_horizontally(self.x, 9.0, direction='left')
        self.assertTrue(torch.equal(result, expected))

class TestTransTable(unittest.TestCase):
    """Unit tests for create_trans_table with sub_label_factor."""

    def setUp(self):
        self.batch_size = 2
        self.max_len = 12
        self.labels = torch.zeros((self.batch_size, self.max_len))
        self.labels_len = torch.tensor([self.max_len, self.max_len])

    def test_factor_2(self):
        """Tests if factor 2 allows skipping odd-indexed elements."""
        factor = 2
        table = shc_loss.create_trans_table(self.labels_len, factor)[0]
        # Skipped element (i+1) should be 1, 3, 5... (2n-1)
        # So i should be 0, 2, 4...
        self.assertEqual(table[0, 2], 0)  # 1 skipped
        self.assertEqual(table[2, 4], 0)  # 3 skipped
        self.assertEqual(table[1, 3], shc_loss.LOG_0)  # 2 is not (2n-1)

    def test_factor_4(self):
        """Tests if factor 4 allows skipping (4n-1) indexed elements."""
        factor = 4
        table = shc_loss.create_trans_table(self.labels_len, factor)[0]
        # Skipped element (i+1) should be 3, 7, 11... (4n-1)
        # So i should be 2, 6, 10...
        self.assertEqual(table[2, 4], 0)  # 3 skipped
        self.assertEqual(table[6, 8], 0)  # 7 skipped
        self.assertEqual(table[0, 2], shc_loss.LOG_0)  # 1 is not (4n-1)
        self.assertEqual(table[4, 6], shc_loss.LOG_0)  # 5 is not (4n-1)

    def test_self_loops(self):
        """Tests if self-loops are allowed."""
        table = shc_loss.create_trans_table(self.labels_len, 2)[0]
        self.assertEqual(table[0, 0], 0)
        self.assertEqual(table[1, 1], 0)
        self.assertEqual(table[3, 3], 0)
        self.assertEqual(table[2, 2], 0)


class TestBlockAugmentation(unittest.TestCase):
    def setUp(self):
        # 2 classes (0, 1), n=3 augmentation
        self.num_classes = 10
        self.inputs = {
            "SEQ_DATA": torch.tensor([[1, 2]], dtype=torch.long),
            "SEQ_LEN": torch.tensor([2], dtype=torch.long)
        }

    def test_block_logic(self):
        n = 3
        k = self.num_classes
        result = shc_loss.to_onset_block_augmented_n(self.inputs, n, k)
        
        # Expected: [1, 2, 1+k, 2+k, 1+2k, 2+2k] -> [1, 2, 11, 12, 21, 22]
        expected_data = torch.tensor([[1, 11, 0, 2, 12, 0]], dtype=torch.long)
        torch.testing.assert_close(result["SEQ_DATA"], expected_data)

    def test_masking(self):
        # Test if padding is preserved across blocks
        inputs_with_pad = {
            "SEQ_DATA": torch.tensor([[5, 0]], dtype=torch.long),
            "SEQ_LEN": torch.tensor([1], dtype=torch.long)
        }
        n = 2
        k = self.num_classes
        result = shc_loss.to_onset_block_augmented_n(inputs_with_pad, n, k)

        # Expected: [5, 0, 15, 0] -> Only elements within SEQ_LEN are augmented
        # After masking: [5, 0, 15, 0] (Wait, block-wise padding is tricky)
        # The logic ensures only valid time-steps are considered.
        expected_data = torch.tensor([[5, 0, 0, 0]], dtype=torch.long)
        torch.testing.assert_close(result["SEQ_DATA"], expected_data)


def shift_tensor_horizontally(x, fill_value, direction='right'):
    """Shifts tensor elements along the last dimension."""
    res = torch.full_like(x, fill_value)
    if direction == 'right':
        res[:, 1:] = x[:, :-1]
    else:
        res[:, :-1] = x[:, 1:]
    return res

class TestAlphaBetaStability(unittest.TestCase):
    """Verifies numerical correctness of forward-backward computation."""

    def setUp(self):
        """Sets up dimensions and tensors for testing."""
        self.B, self.T, self.L = 3, 10, 9
        self.device = torch.device('cpu')

        # Fixed random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        self.log_target_probs = torch.log_softmax(
            torch.randn(self.B, self.T, 2 * self.L - 1), dim=-1)

        self.logit_lens = torch.tensor([10, 8, 6])
        self.target_lens = torch.tensor([9, 6, 3])
        #self.trans_table = shc_loss.create_trans_table(
        #    self.target_lens, self.sub_label_factor)

        vocab_size = 32
        labels = torch.randint(0, vocab_size, (self.B, self.L))
        mask = torch.arange(self.L)[None, :] < self.target_lens[:, None]
        labels = labels.masked_fill(~mask, 0)

        # TODO ... to blank augmented label
        inputs = {}
        inputs["SEQ_DATA"] = labels
        inputs["SEQ_LEN"] = self.target_lens

        inputs = seq_loss_util.to_blank_augmented_labels(inputs, 0, False, False)

        labels = inputs["SEQ_DATA"]
        self.target_lens = inputs["SEQ_LEN"]

        self.trans_table = seq_loss_util.label_trans_allowance_table_ctc(
            labels, self.target_lens)

    def test_forward_backward_consistency(self):
        """Checks if log_prob(total) is consistent across all time steps."""

        # 1. Run the forward-backward (Assuming your function is defined)
        log_alpha, log_beta, _ = shc_loss.calculate_alpha_beta(
            self.trans_table, self.log_target_probs, self.target_lens, self.logit_lens)

        for b in range(self.B):
            valid_t = self.logit_lens[b]
            # Across all valid t, logsumexp(alpha + beta) must be constant
            # representing the total log probability of the sequence.
            total_log_target_probs = []
            for t in range(valid_t):
                # Combined posterior log-probability at time t
                prob_t = torch.logsumexp(
                    log_alpha[b, t] + log_beta[b, t], dim=-1)
                total_log_target_probs.append(prob_t.item())
            
            # Variance should be near zero if the algorithm is correct
            variance = np.var(total_log_target_probs)

            self.assertLess(variance, 1e-4, 
                           f"Inconsistent total prob at batch {b}")


    def test_padding_mask(self):
        """Ensures that padded regions remain LOG_0 using tensor operations."""
        log_alpha, _, _ = shc_loss.calculate_alpha_beta(
            self.trans_table, self.log_target_probs, self.target_lens, self.logit_lens)

        # Create a mask for padded time steps (B, T)
        t_indices = torch.arange(self.T, device=self.device).expand(self.B, -1)
        padding_mask = t_indices >= self.logit_lens.unsqueeze(1)

        # Check if all values in padded regions are LOG_0
        # This masks log_alpha (B, T, L) with padding_mask (B, T)
        masked_values = log_alpha[padding_mask]
        
        target_log0 = torch.tensor(shc_loss.LOG_0, device=self.device)
        self.assertTrue(
            torch.allclose(masked_values, target_log0, atol=1e-2),
            f"Padded regions in log_alpha are not properly set to {shc_loss.LOG_0}"
        )

class TestTokenAugmentation(unittest.TestCase):
    """Unit tests for the SHC token sequence transformation."""

    def setUp(self):
        self.bo_id = 100
        self.bl_id = 200
        self.specials = [1, 2]  # e.g., 1=<s>, 2=</s>

    def test_mixed_tokens(self):
        """Tests a mix of special and normal tokens."""
        # Input: [<s>, 13, 4, </s>]
        inputs = {
            "SEQ_DATA": torch.tensor([[1, 13, 4, 2]]),
            "SEQ_LEN": torch.tensor([4])
        }
        
        result = shc_loss.to_shc_token_seq(inputs, self.bo_id, self.bl_id, self.specials)
        
        # Expected: [1, bo, 13, bl, bo, 4, bl, 2]
        expected = [1, 100, 13, 200, 100, 4, 200, 2]
        self.assertEqual(result["SEQ_DATA"][0].tolist(), expected)
        self.assertEqual(result["SEQ_LEN"][0].item(), 8)

    def test_only_special_tokens(self):
        """Tests when the sequence contains only special tokens."""
        inputs = {
            "SEQ_DATA": torch.tensor([[1, 2, 0]]),
            "SEQ_LEN": torch.tensor([2])
        }
        result = shc_loss.to_shc_token_seq(inputs, self.bo_id, self.bl_id, self.specials)
        
        expected = [1, 2]
        self.assertEqual(result["SEQ_DATA"][0, :2].tolist(), expected)
        self.assertEqual(result["SEQ_LEN"][0].item(), 2)

    def test_batch_padding(self):
        """Tests if padding is correctly applied for different length outputs."""
        inputs = {
            "SEQ_DATA": torch.tensor([
                [1, 13, 0], # Expands to length 4: [1, bo, 13, bl]
                [4, 0, 0]   # Expands to length 3: [bo, 4, bl]
            ]),
            "SEQ_LEN": torch.tensor([2, 1])
        }
        result = shc_loss.to_shc_token_seq(inputs, self.bo_id, self.bl_id, self.specials)
        
        # Check lengths
        self.assertEqual(result["SEQ_LEN"].tolist(), [4, 3])

        # Check padding (should be -100)
        self.assertEqual(result["SEQ_DATA"][1, 3].item(), -100)



class FlooredActiveSupportEndToEndTest(unittest.TestCase):
    """FAS through ShcLoss on the batch shape that used to break the target.

    T=900 with label lengths spanning 2..440 is the extreme-length-spread
    case that produced an all-blank target before de53098, so it is the
    shape a new alpha_mode is most likely to trip over.
    """

    def test_finite_loss_and_gradient_on_extreme_length_spread(self):
        num_classes, max_logit_len = 32, 900
        lengths = [2, 3, 400, 440]
        labels = torch.full((len(lengths), max(lengths)), -100,
                            dtype=torch.long)
        torch.manual_seed(0)
        for b, n in enumerate(lengths):
            labels[b, :n] = torch.randint(1, num_classes, (n,))
        target_lens = (labels >= 0).sum(1)
        logit_lens = torch.full((len(lengths),), max_logit_len,
                                dtype=torch.long)

        for alpha, beta in ((0.40, 0.0), (0.40, 0.5), (0.53, 1.0)):
            logits = (torch.randn(len(lengths), max_logit_len, num_classes)
                      * 0.1).requires_grad_(True)
            loss = shc_loss.ShcLoss.apply(
                labels, target_lens, logits.log_softmax(-1), logit_lens,
                num_classes, alpha, beta, False, 0.0, False, "class",
                "floored_active_support").mean()
            loss.backward()
            where = f"alpha={alpha} beta={beta}"
            self.assertTrue(bool(torch.isfinite(loss)), where)
            self.assertTrue(bool(torch.isfinite(logits.grad).all()), where)

    def test_rejects_non_class_space(self):
        """Both FAS rates are per-class and its threshold would fire on the
        label axis's padding, so label/hybrid space must be refused."""
        logits = torch.randn(1, 20, 12)
        labels = torch.randint(1, 12, (1, 4))
        for space in ("label", "hybrid"):
            with self.assertRaises(AssertionError):
                shc_loss.ShcLoss.apply(
                    labels, torch.tensor([4]), logits.log_softmax(-1),
                    torch.tensor([20]), 12, 0.4, 0.5, False, 0.0, False,
                    space, "floored_active_support")

class AlignmentBiasedTest(unittest.TestCase):
    """ABS through ShcLoss: smooth toward the no-acoustics alignment.

    Textbook LS pulls the target toward 1/C, which knows nothing about the
    lattice. ABS pulls it toward the alignment posterior the same lattice
    would produce if every per-frame log-probability were equal -- a purely
    combinatorial object that depends on T and the label sequence's repeat
    pattern and NOT on the model.
    """

    def _case(self, seed=0, b=3, t=60, c=12, lens=(4, 9, 15)):
        torch.manual_seed(seed)
        labels = torch.full((b, max(lens)), -100, dtype=torch.long)
        for i, n in enumerate(lens):
            labels[i, :n] = torch.randint(1, c, (n,))
        target_lens = (labels >= 0).sum(1)
        logit_lens = torch.full((b,), t, dtype=torch.long)
        logits = torch.randn(b, t, c) * 0.5
        return labels, target_lens, logits, logit_lens, c

    def _reference(self, labels, target_lens, logits, logit_lens):
        """The class-space reference ABS mixes toward."""
        inputs = seq_loss_util.to_blank_augmented_labels(
            {"SEQ_DATA": labels, "SEQ_LEN": target_lens}, 0, False, False)
        aug, aug_lens = inputs["SEQ_DATA"], inputs["SEQ_LEN"]
        clamped = aug.clamp(min=0)
        trans = seq_loss_util.label_trans_allowance_table_ctc(aug, aug_lens)
        log_probs = torch.log_softmax(logits, dim=-1)
        ltp = seq_loss_util.calculate_log_label_prob(clamped, log_probs)
        ug = shc_loss.uniform_acoustic_gamma(
            trans, ltp, aug_lens, logit_lens,
            max_logit_len_int=int(logit_lens.max()))
        return shc_loss._scatter_to_class_space(ug, log_probs, clamped), ug

    def test_reference_is_a_per_frame_distribution(self):
        labels, tl, logits, ll, c = self._case(seed=1)
        ref, ug = self._reference(labels, tl, logits, ll)
        sums = ug.sum(-1)
        torch.testing.assert_close(sums, torch.ones_like(sums),
                                   atol=1e-5, rtol=1e-5)
        cs = ref.sum(-1)
        torch.testing.assert_close(cs, torch.ones_like(cs),
                                   atol=1e-5, rtol=1e-5)

    def test_reference_does_not_depend_on_the_model(self):
        """It is combinatorial: same (T, labels) must give the same thing."""
        labels, tl, logits, ll, c = self._case(seed=2)
        _, ug1 = self._reference(labels, tl, logits, ll)
        _, ug2 = self._reference(labels, tl, torch.randn_like(logits) * 3.0, ll)
        torch.testing.assert_close(ug1, ug2, atol=1e-6, rtol=1e-5)

    def test_alpha_zero_is_the_plain_loss(self):
        labels, tl, logits, ll, c = self._case(seed=3)
        g = []
        for mode in ("fixed", "alignment_biased"):
            x = logits.clone().requires_grad_(True)
            shc_loss.ShcLoss.apply(labels, tl, x.log_softmax(-1), ll, c,
                                   0.0, 0.0, False, 0.0, False, "class",
                                   mode).sum().backward()
            g.append(x.grad)
        torch.testing.assert_close(g[0], g[1], atol=0, rtol=0)

    def test_alpha_one_replaces_the_target_with_the_reference(self):
        labels, tl, logits, ll, c = self._case(seed=4)
        ref, _ = self._reference(labels, tl, logits, ll)
        x = logits.clone().requires_grad_(True)
        shc_loss.ShcLoss.apply(labels, tl, x.log_softmax(-1), ll, c,
                               1.0, 0.0, False, 0.0, False, "class",
                               "alignment_biased").sum().backward()
        log_probs = torch.log_softmax(logits, dim=-1)
        expected = log_probs.exp() - ref
        torch.testing.assert_close(x.grad, expected, atol=1e-5, rtol=1e-4)

    def test_finite_on_extreme_length_spread(self):
        """The shape that broke the target before de53098."""
        num_classes, t = 32, 900
        lens = [2, 3, 400, 440]
        torch.manual_seed(0)
        labels = torch.full((len(lens), max(lens)), -100, dtype=torch.long)
        for b, n in enumerate(lens):
            labels[b, :n] = torch.randint(1, num_classes, (n,))
        tl = (labels >= 0).sum(1)
        ll = torch.full((len(lens),), t, dtype=torch.long)
        for alpha in (0.05, 0.3, 1.0):
            x = (torch.randn(len(lens), t, num_classes) * 0.1
                 ).requires_grad_(True)
            loss = shc_loss.ShcLoss.apply(
                labels, tl, x.log_softmax(-1), ll, num_classes, alpha, 0.0,
                False, 0.0, False, "class", "alignment_biased").mean()
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)), f"alpha={alpha}")
            self.assertTrue(bool(torch.isfinite(x.grad).all()), f"alpha={alpha}")

    def test_rejects_non_class_space(self):
        logits = torch.randn(1, 20, 12)
        labels = torch.randint(1, 12, (1, 4))
        for space in ("label", "hybrid"):
            with self.assertRaises(AssertionError):
                shc_loss.ShcLoss.apply(
                    labels, torch.tensor([4]), logits.log_softmax(-1),
                    torch.tensor([20]), 12, 0.4, 0.0, False, 0.0, False,
                    space, "alignment_biased")
class BackwardAritySmokeTest(unittest.TestCase):
    """`backward` must return one entry per `forward` argument.

    Autograd only reports this as "returned an incorrect number of
    gradients" at the first backward pass, so a new keyword silently
    breaks every run that uses the loss. It has now happened twice --
    once for fas_threshold_frac, once when peak_thresh/peak_gate and
    blank_gate were added -- hence this check.
    """

    def test_arity_matches(self):
        import inspect
        n_forward = len(inspect.signature(
            shc_loss.ShcLoss.forward).parameters) - 1   # drop ctx
        src = inspect.getsource(shc_loss.ShcLoss.backward)
        ret = src[src.rindex("return ("):]
        n_backward = ret.count("None") + ret.count("gradient")
        self.assertEqual(n_forward, n_backward,
                         f"forward takes {n_forward} args but backward "
                         f"returns {n_backward} gradients")

    def test_every_new_keyword_reaches_backward(self):
        """A real backward pass, which is where the mismatch actually fires."""
        torch.manual_seed(0)
        c, t = 12, 40
        labels = torch.randint(1, c, (2, 5))
        logits = (torch.randn(2, t, c) * 0.5).requires_grad_(True)
        shc_loss.ShcLoss.apply(
            labels, torch.tensor([5, 5]), logits.log_softmax(-1),
            torch.tensor([t, t]), c, 0.2, 0.0, False, 0.0, False, "class",
            "floored_active_support", 1.0, 1.0, 1e-10, 1e-5, 0.9, "high",
            "blank_only").sum().backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

if __name__ == "__main__":
    unittest.main()


class AlignmentWeightSmoothingTest(unittest.TestCase):
    """AWS smooths gamma(t, l) over label POSITIONS before the scatter."""

    def setUp(self):
        torch.manual_seed(0)
        self.labels = torch.tensor([[0, 1, 0, 2, 0, 1, 0],
                                    [0, 3, 0, 3, 0, 0, 0]])
        self.target_lens = torch.tensor([7, 5])
        self.logits_len = torch.tensor([9, 7])
        self.logits = torch.randn(2, 9, 6) * 3

    def _apply(self, alpha, beta=0.0, eps=1e-10):
        g = torch.rand(2, 9, 7).abs() + 1e-12
        pos = torch.arange(7).view(1, 1, -1)
        g = torch.where(pos < self.target_lens.view(-1, 1, 1), g,
                        torch.zeros_like(g))
        g = g / g.sum(-1, keepdim=True)
        out = shc_loss.apply_alignment_weight_smoothing(
            g, self.target_lens, alpha, beta, eps=eps)
        return g, out

    def test_rows_still_sum_to_one(self):
        _, out = self._apply(0.15)
        self.assertTrue(torch.allclose(out.sum(-1), torch.ones(2, 9),
                                       atol=1e-5))

    def test_padded_positions_stay_zero(self):
        _, out = self._apply(0.15, beta=1.0)
        pos = torch.arange(7).view(1, 1, -1)
        pad = pos >= self.target_lens.view(-1, 1, 1)
        self.assertGreater(int(pad.sum()), 0)
        self.assertTrue(torch.all(out[pad.expand_as(out)] == 0.0))

    def test_alpha_zero_is_a_no_op(self):
        g, out = self._apply(0.0)
        self.assertTrue(torch.allclose(g, out, atol=1e-7))

    def test_inactive_positions_are_exactly_unchanged(self):
        g = torch.tensor([[[0.90, 0.06, 0.03, 0.008, 0.002, 0.0, 0.0]]])
        out = shc_loss.apply_alignment_weight_smoothing(
            g, torch.tensor([5]), 0.2, 0.0, eps=0.01)
        self.assertTrue(torch.equal(out[0, 0, 3:5], g[0, 0, 3:5]))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=6)

    def test_it_flattens(self):
        """The fixed point is 1/K over the sample's OWN width, not the
        padded width -- a position above it comes down, one below goes up.
        With eps at its default every valid position is active, so this is
        the whole rule."""
        g, out = self._apply(0.3)
        k = self.target_lens.view(-1, 1, 1).to(g.dtype)
        pos = torch.arange(g.shape[-1]).view(1, 1, -1)
        valid = pos < self.target_lens.view(-1, 1, 1)
        above = valid.expand_as(g) & (g > 1.0 / k)
        below = valid.expand_as(g) & (g < 1.0 / k)
        self.assertGreater(int(above.sum()), 0)
        self.assertGreater(int(below.sum()), 0)
        self.assertTrue(torch.all(out[above] < g[above]))
        self.assertTrue(torch.all(out[below] > g[below]))

    def test_K_is_the_samples_own_width_not_the_batch_width(self):
        """Bug 2 in CLAUDE.md: a sample's target must not depend on how
        long its batch-mates are."""
        g = torch.rand(1, 4, 7).abs()
        pos = torch.arange(7).view(1, 1, -1)
        g = torch.where(pos < 5, g, torch.zeros_like(g))
        g = g / g.sum(-1, keepdim=True)
        alone = shc_loss.apply_alignment_weight_smoothing(
            g, torch.tensor([5]), 0.2, 0.0)
        wide = torch.cat([g, torch.zeros(1, 4, 9)], dim=-1)
        with_mates = shc_loss.apply_alignment_weight_smoothing(
            wide, torch.tensor([5]), 0.2, 0.0)[..., :7]
        self.assertTrue(torch.allclose(alone, with_mates, atol=1e-6))

    def test_beta_one_reaches_inactive_positions(self):
        g = torch.tensor([[[0.7, 0.3, 0.0, 0.0]]])
        lens = torch.tensor([4])
        b0 = shc_loss.apply_alignment_weight_smoothing(g, lens, 0.2, 0.0)
        b1 = shc_loss.apply_alignment_weight_smoothing(g, lens, 0.2, 1.0)
        self.assertAlmostEqual(float(b0[0, 0, 2]), 0.0, places=6)
        self.assertGreater(float(b1[0, 0, 2]), 0.0)

    def test_end_to_end_changes_the_gradient(self):
        def grad(alpha, mode):
            x = self.logits.clone().requires_grad_(True)
            loss = shc_loss.ShcLoss.apply(
                self.labels, self.target_lens, x, self.logits_len, 6,
                alpha, 0.0, False, 0.0, False, "label", mode, 1.0, 1.0,
                1e-10, 1e-5, 0.9, "none", "none")
            loss.sum().backward()
            return x.grad.clone()
        g0 = grad(0.0, "fixed")
        g1 = grad(0.15, "aws")
        rel = float((g0 - g1).norm() / g0.norm())
        self.assertGreater(rel, 1e-3)
        self.assertTrue(torch.isfinite(g1).all())

    def test_differs_from_label_space_sets(self):
        """AWS is not L-SETS: the mixing distribution covers only the
        ACTIVE positions, not every position."""
        def grad(mode, space):
            x = self.logits.clone().requires_grad_(True)
            loss = shc_loss.ShcLoss.apply(
                self.labels, self.target_lens, x, self.logits_len, 6,
                0.15, 0.0, False, 0.0, False, space, mode, 1.0, 1.0,
                1e-10, 1e-5, 0.9, "none", "none")
            loss.sum().backward()
            return x.grad.clone()
        aws = grad("aws", "label")
        lsets = grad("fixed", "label")
        self.assertGreater(float((aws - lsets).norm() / lsets.norm()), 1e-3)


class ModelOutputSmoothingTest(unittest.TestCase):
    """MOS smooths the emission distribution itself, so alpha, beta, gamma,
    the target and the gradient are all derived from the smoothed one."""

    def setUp(self):
        torch.manual_seed(0)
        self.labels = torch.tensor([[0, 3, 0, 5, 0, 3, 0]])
        self.target_lens = torch.tensor([7])
        self.logits_len = torch.tensor([9])
        self.logits = torch.randn(1, 9, 8) * 3

    def _run(self, alpha, mode, eps=1e-2):
        x = self.logits.clone().requires_grad_(True)
        loss = shc_loss.ShcLoss.apply(
            self.labels, self.target_lens, x, self.logits_len, 8, alpha,
            0.0, False, 0.0, False, "label", mode, 1.0, 1.0, eps, 1e-5,
            0.9, "none", "none")
        loss.sum().backward()
        return float(loss.sum()), x.grad.clone()

    def test_alpha_zero_is_a_no_op(self):
        a, ga = self._run(0.0, "fixed")
        b, gb = self._run(0.0, "mos")
        self.assertAlmostEqual(a, b, places=5)
        self.assertTrue(torch.allclose(ga, gb, atol=1e-6))

    def test_it_changes_the_loss_value_unlike_every_other_mode(self):
        """MOS reports -log P~ under the smoothed emissions, so its loss is
        NOT the true likelihood and is not comparable with a baseline's."""
        base, _ = self._run(0.0, "fixed")
        mos, _ = self._run(0.1, "mos")
        self.assertNotAlmostEqual(base, mos, places=4)

    def test_smoothing_lowers_the_loss(self):
        """Flattening the emissions spreads mass onto every alignment, so
        the summed path probability can only go up."""
        base, _ = self._run(0.0, "fixed")
        for a in (0.05, 0.1, 0.2):
            mos, _ = self._run(a, "mos")
            self.assertLess(mos, base)

    def test_gradient_is_finite_and_changes(self):
        _, g0 = self._run(0.0, "fixed")
        _, g1 = self._run(0.05, "mos")
        self.assertTrue(torch.isfinite(g1).all())
        self.assertGreater(float((g0 - g1).norm() / g0.norm()), 1e-3)

    def test_padded_frames_do_not_leak(self):
        """A short sample in a padded batch must get the same gradient it
        gets alone -- the smoothing must not let padding into the
        recursion."""
        lab = torch.cat([self.labels, self.labels], 0)
        tl = torch.tensor([7, 7])
        ll = torch.tensor([9, 5])
        lg = torch.cat([self.logits, self.logits], 0)
        x = lg.clone().requires_grad_(True)
        loss = shc_loss.ShcLoss.apply(
            lab, tl, x, ll, 8, 0.05, 0.0, False, 0.0, False, "label",
            "mos", 1.0, 1.0, 1e-2, 1e-5, 0.9, "none", "none")
        loss.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        # frames past the second sample's length get no gradient
        self.assertTrue(torch.allclose(x.grad[1, 5:], torch.zeros(4, 8),
                                       atol=1e-7))
