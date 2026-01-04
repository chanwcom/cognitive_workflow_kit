import torch
import unittest


def adaptive_label_smoothing(
    estimated_targets: torch.Tensor,
    model_outputs: torch.Tensor,
    alpha: float,
    eps: float = 1e-10
) -> torch.Tensor:
    """Performs batch-wise adaptive label smoothing.

    Args:
        estimated_targets: Soft target sequence of shape [B, T, C].
        model_outputs: Model probabilities (Softmax) of shape [B, T, C].
        alpha: Base smoothing coefficient.
        eps: Threshold to identify active classes in targets.

    Returns:
        torch.Tensor: Smoothed targets of shape [B, T, C].
    """
    assert estimated_targets.shape == model_outputs.shape, "Shape mismatch."

    # Identify active indices: [B, T, C]
    mask = (estimated_targets > eps).to(model_outputs.dtype)

    # Calculate sum of probabilities for active classes: [B, T, 1]
    p_step = torch.sum(model_outputs * mask, dim=-1, keepdim=True)

    # Average confidence across time steps per batch: [B, 1, 1]
    p_batch = torch.mean(p_step, dim=1, keepdim=True)

    # Compute shared weight w for the entire sequence: [B, 1, 1]
    w = alpha + (1.0 - alpha) * p_batch

    # Bound the adaptive weight w.
    w = torch.clamp(w, min = 0.0, max=1.0)

    # Apply smoothing using batch-consistent weight w
    num_classes = estimated_targets.size(-1)
    c_inv = 1.0 / num_classes
    smoothed_targets = (w * estimated_targets) + ((1.0 - w) * c_inv)

    return smoothed_targets


def label_smoothing(
    estimated_targets: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """Computes label smoothing using a fixed alpha coefficient.

    This function applies standard label smoothing by performing a linear
    interpolation between the input targets and a uniform distribution.
    The formula used is:
        smoothed_targets = (alpha * targets) + ((1.0 - alpha) / num_classes)

    Args:
        estimated_targets: Target tensor (one-hot or soft targets) of shape [..., C].
        alpha: Smoothing coefficient where 1.0 preserves the original targets
            and lower values increase the smoothing effect.

    Returns:
        torch.Tensor: Smoothed target tensor of the same shape as input.
    """
    num_classes = estimated_targets.size(-1)
    uniform_distribution = 1.0 / num_classes
    
    # Standard label smoothing formula
    return (alpha * estimated_targets) + ((1.0 - alpha) * uniform_distribution)


class TestLabelSmoothing(unittest.TestCase):
    """Unit tests for standard label smoothing."""

    def setUp(self):
        """Sets up the default alpha for testing."""
        self.alpha = 0.8

    def test_smoothing_logic(self):
        """Verifies that the smoothing formula is applied correctly."""
        # Case: B=1, T=1, C=3 (Single step)
        e_targets = torch.tensor([[[1.0, 0.0, 0.0]]])
        
        # Expected: 0.8 * [1, 0, 0] + 0.2 * [1/3, 1/3, 1/3]
        # = [0.8 + 0.0666..., 0.0666..., 0.0666...]
        c_inv = 1.0 / 3.0
        expected = torch.tensor([[[
            0.8 + 0.2 * c_inv,
            0.2 * c_inv,
            0.2 * c_inv
        ]]])

        result = label_smoothing(e_targets, self.alpha)
        torch.testing.assert_close(result, expected)

    def test_shape_consistency(self):
        """Ensures the output shape matches the input shape."""
        shapes = [(2, 10, 5), (1, 3), (4, 4, 4, 8)]
        for shape in shapes:
            with self.subTest(shape=shape):
                dummy_input = torch.randn(*shape)
                result = label_smoothing(dummy_input, self.alpha)
                self.assertEqual(result.shape, dummy_input.shape)

    def test_zero_alpha(self):
        """Verifies that alpha=0.0 results in a pure uniform distribution."""
        e_targets = torch.tensor([1.0, 0.0, 0.0, 0.0]) # C=4
        result = label_smoothing(e_targets, alpha=0.0)
        expected = torch.full_like(e_targets, 0.25)
        torch.testing.assert_close(result, expected)


class TestAdaptiveLabelSmoothing(unittest.TestCase):
    """Unit tests for batch-consistent adaptive label smoothing."""

    def setUp(self):
        self.alpha = 0.2

    def test_batch_wise_consistency(self):
        """Verifies that smoothing weight is consistent across time steps."""
        # B=1, T=2, C=3
        e_targets = torch.tensor([[
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0]
        ]])
        m_outputs = torch.tensor([[
            [0.8, 0.1, 0.1],  # T0: active p = 0.8
            [0.4, 0.3, 0.3]   # T1: active p = 0.4 + 0.3 = 0.7
        ]])

        # Average p = (0.8 + 0.7) / 2 = 0.75
        # w = 0.2 + (0.8 * 0.75) = 0.8
        # T0 Expected: 0.8 * [1, 0, 0] + 0.2 * [1/3, 1/3, 1/3]
        # T1 Expected: 0.8 * [0.5, 0.5, 0] + 0.2 * [1/3, 1/3, 1/3]
        c_inv = 1.0 / 3.0
        expected = torch.tensor([[
            [0.8 + 0.2 * c_inv, 0.2 * c_inv, 0.2 * c_inv],
            [0.4 + 0.2 * c_inv, 0.4 + 0.2 * c_inv, 0.2 * c_inv]
        ]])

        result = adaptive_label_smoothing(e_targets, m_outputs, self.alpha)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
