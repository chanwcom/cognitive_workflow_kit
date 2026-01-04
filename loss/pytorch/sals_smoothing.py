import torch
import unittest


def adaptive_label_smoothing(
    estimated_targets: torch.Tensor,
    model_outputs: torch.Tensor,
    alpha: float,
    eps: float = 1e-10
) -> torch.Tensor:
    """Performs adaptive label smoothing per batch and time step.

    Args:
        estimated_targets: Soft target sequence of shape [B, T, C].
        model_outputs: Model probabilities (Softmax) of shape [B, T, C].
        alpha: Base smoothing coefficient.
        eps: Threshold to identify active classes in estimated_targets.

    Returns:
        torch.Tensor: Smoothed targets of shape [B, T, C].
    """
    assert estimated_targets.shape == model_outputs.shape, "Shape mismatch."

    # Identify 'active' indices: shape [B, T, C]
    mask = (estimated_targets > eps).to(model_outputs.dtype)

    # Sum model probabilities for active indices: shape [B, T, 1]
    p = torch.sum(model_outputs * mask, dim=-1, keepdim=True)

    # Calculate unique weight w for every (batch, time): shape [B, T, 1]
    w = alpha + (1.0 - alpha) * p

    # Apply adaptive smoothing: w * target + (1 - w) * (1/C)
    num_classes = estimated_targets.size(-1)
    c_inv = 1.0 / num_classes
    smoothed_targets = (w * estimated_targets) + ((1.0 - w) * c_inv)

    return smoothed_targets


class TestAdaptiveLabelSmoothing(unittest.TestCase):
    """Unit tests using direct tensor comparison and 80-char width."""

    def setUp(self):
        self.alpha = 0.2
        self.eps = 1e-10

    def test_tensor_equality_logic(self):
        """Validates the output against a pre-calculated expected tensor."""
        # Setup: B=1, T=2, C=3
        # T0: Single active class (index 0)
        # T1: Multi active class (indices 0, 1)
        e_targets = torch.tensor([[
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0]
        ]])

        m_outputs = torch.tensor([[
            [0.8, 0.1, 0.1],  # T0: p = 0.8
            [0.4, 0.3, 0.3]   # T1: p = 0.7
        ]])

        # Manual Calculation for Expected:
        # T0: w = 0.2 + (0.8 * 0.8) = 0.84
        #     val = 0.84 * [1, 0, 0] + 0.16 * [1/3, 1/3, 1/3]
        #     val = [0.84 + 0.05333, 0.05333, 0.05333] = [0.89333, 0.05333, ...]
        # T1: w = 0.2 + (0.8 * 0.7) = 0.76
        #     val = 0.76 * [0.5, 0.5, 0] + 0.24 * [1/3, 1/3, 1/3]
        #     val = [0.38 + 0.08, 0.38 + 0.08, 0.08] = [0.46, 0.46, 0.08]

        c_inv = 1.0 / 3.0
        expected_smoothed = torch.tensor([[
            [0.84 + 0.16 * c_inv, 0.16 * c_inv, 0.16 * c_inv],
            [0.38 + 0.24 * c_inv, 0.38 + 0.24 * c_inv, 0.24 * c_inv]
        ]])

        result = adaptive_label_smoothing(e_targets, m_outputs, self.alpha)

        # Direct tensor comparison
        torch.testing.assert_close(result, expected_smoothed)

    def test_zero_target_handling(self):
        """Ensures logic holds when no classes exceed eps."""
        e_targets = torch.zeros((1, 1, 3))
        m_outputs = torch.tensor([[[0.3, 0.3, 0.4]]])

        # p = 0.0, w = alpha (0.2)
        # expected = 0.2 * 0 + 0.8 * (1/3) = 0.2666...
        val = 0.8 / 3.0
        expected = torch.tensor([[[val, val, val]]])

        result = adaptive_label_smoothing(e_targets, m_outputs, self.alpha)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
