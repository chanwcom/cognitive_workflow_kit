import torch
import unittest

# alpha가 0.2 일때 optimal 성능 보임.
# 0.97인 LS보다 살짝 더 좋음.: 20.243 %
# 20.426%
# 
def adaptive_label_smoothing(
    estimated_targets: torch.Tensor,
    model_outputs: torch.Tensor,
    alpha: float,
    eps: float = 1e-10
) -> torch.Tensor:
    """Performs batch-wide label smoothing by matching entropy levels.

    Calculates the average entropy difference between targets and model
    outputs, then mixes a uniform distribution into the targets to bridge
    the gap, ensuring consistent smoothing across the entire batch.

    Args:
        estimated_targets: Soft targets of shape [B, T, C].
        model_outputs: Model probabilities (Softmax) of shape [B, T, C].
        eps: Small constant for numerical stability in log.

    Returns:
        torch.Tensor: Smoothed targets of shape [B, T, C].
    """
    assert estimated_targets.shape == model_outputs.shape, "Shape mismatch."

    # Compute mean entropy for targets and model outputs over [B, T].
    # H = -sum(p * log(p))
    h_target = -torch.mean(
        torch.sum(estimated_targets * torch.log(estimated_targets + eps), dim=-1)
    )
    h_model = -torch.mean(
        torch.sum(model_outputs * torch.log(model_outputs + eps), dim=-1)
    )

    # Calculate required entropy gap to fill (only if model is more confident).
    diff = torch.clamp(alpha * (h_model - h_target), min=0.0)

    # Maximum possible entropy (Uniform distribution).
    num_classes = estimated_targets.size(-1)
    h_max = torch.log(torch.tensor(float(num_classes), device=estimated_targets.device))

    # Calculate global smoothing weight 'w' based on the entropy gap.
    # w=1 means no smoothing, w=0 means pure uniform distribution.
    denom = torch.clamp(h_max - h_target, min=eps)
    intensity = torch.clamp(diff / denom, max=1.0)

    w = 1.0 - intensity
    w = torch.clamp(w, min=0.9, max=1.0)

    # Apply uniform smoothing across all samples and time steps.
    c_inv = 1.0 / num_classes
    smoothed_targets = (w * estimated_targets) + ((1.0 - w) * c_inv)

    return smoothed_targets


def label_smoothing(
    estimated_targets: torch.Tensor,
    alpha: float,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Applies label smoothing only to active alignment indices.

    Args:
        estimated_targets: Target tensor of shape [..., C].
        alpha: Smoothing coefficient (1.0 = no smoothing).
        eps: Small value to avoid division by zero.

    Returns:
        A smoothed tensor of the same shape as input.
    """
    # Create mask for valid alignment paths: [B, T, C]
    mask = (estimated_targets > eps).to(estimated_targets.dtype)

    # Count active classes per timestep: [B, T, 1]
    active_counts = mask.sum(dim=-1, keepdim=True)

    # Distribute smoothing mass within active indices only
    # Avoids putting probability on impossible tokens
    uniform_restricted = mask / (active_counts + eps)

    # Blend original targets with restricted uniform distribution
    return (alpha * estimated_targets) + ((1.0 - alpha) * uniform_restricted)

def label_smoothing_old(
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
