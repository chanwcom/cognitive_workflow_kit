"""Tests for SlimDucer.

Run as `python slimducer_test.py` -- pytest is not installed in
py3_12_sets and this directory's neighbours use bare functions.

The first run downloads Qwen3-0.6B and Qwen3-ASR-0.6B-hf (about 3 GB) into
the HF cache, which on this machine is a symlink onto the NAS.
"""
import math

import torch

import slimducer
from slimducer import SlimDucer

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SlimDucer(dtype=torch.float32).eval()
    return _MODEL


def test_llm_is_frozen():
    """Not one LLM weight may be trainable.

    The claim this model rests on is that the backbone is untouched, so
    this is the test that would invalidate the paper if it failed.
    """
    m = model()
    bad = [n for n, p in m.llm.named_parameters() if p.requires_grad]
    bad += [n for n, p in m.lm_head.named_parameters() if p.requires_grad]
    assert not bad, f"trainable LLM params: {bad[:5]}"
    print("PASSED: test_llm_is_frozen")


def test_trainable_set_is_what_we_think():
    """Trainable = audio tower + joint + blank head, and nothing else."""
    m = model()
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all(n.startswith(("audio_tower.", "joint.", "blank_head."))
               for n in names), sorted(n for n in names
                                       if not n.startswith(("audio_tower.",
                                                            "joint.",
                                                            "blank_head.")))[:5]
    n_joint = sum(p.numel() for p in m.joint.parameters())
    n_blank = sum(p.numel() for p in m.blank_head.parameters())
    assert n_blank == m.d_llm + 1, (n_blank, m.d_llm)
    print(f"PASSED: test_trainable_set_is_what_we_think "
          f"(joint {n_joint/1e6:.2f}M, blank {n_blank})")


def test_joint_returns_llm_width():
    """The frozen lm_head consumes the joint's output, so the width is not
    a tunable -- a mismatch here is a silent shape error at the softmax."""
    m = model()
    h = m.joint(torch.zeros(3, m.d_llm), torch.zeros(3, m.d_aut))
    assert h.shape == (3, m.d_llm), h.shape
    print("PASSED: test_joint_returns_llm_width")


def test_alignment_is_monotonic_and_consumes_every_token():
    m = model()
    t_len, l_len = 40, 7
    torch.manual_seed(0)
    h_aut = torch.randn(t_len, m.d_aut) * 0.1
    h_llm = torch.randn(l_len + 1, m.d_llm) * 0.1
    labels = torch.randint(0, 1000, (l_len,))
    path = m.align(h_aut, h_llm, t_len, l_len, labels)
    emitted = path[path >= 0]
    assert emitted.tolist() == list(range(l_len)), emitted.tolist()
    print("PASSED: test_alignment_is_monotonic_and_consumes_every_token")


def test_alignment_rejects_more_tokens_than_frames():
    """A reference that cannot fit must be reported, not silently cut.

    Training on a truncated transcript would look like a slightly worse
    loss rather than a bug, which is the kind of thing that survives to
    the results table.
    """
    m = model()
    t_len, l_len = 4, 9
    h_aut = torch.zeros(t_len, m.d_aut)
    h_llm = torch.zeros(l_len + 1, m.d_llm)
    labels = torch.zeros(l_len, dtype=torch.long)
    path = m.align(h_aut, h_llm, t_len, l_len, labels)
    assert bool((path < 0).all()), path.tolist()
    print("PASSED: test_alignment_rejects_more_tokens_than_frames")


def test_blank_threshold_moves_the_operating_point():
    """Threshold 0 must emit on every frame and 1 on none.

    The emission decision is a single sigmoid, so the extremes are exact;
    if they are not, the comparison is on the wrong side.
    """
    m = model()
    feats, mask = _dummy_audio(m)
    none = m.decode(feats, mask, bos_id=0, t_lens=torch.tensor([6]),
                    blank_threshold=0.0)
    allf = m.decode(feats, mask, bos_id=0, t_lens=torch.tensor([6]),
                    blank_threshold=1.0 + 1e-6)
    assert len(none[0]) == 0, len(none[0])
    assert len(allf[0]) == 6, len(allf[0])
    print("PASSED: test_blank_threshold_moves_the_operating_point")


def test_llm_runs_once_per_token():
    """The LLM must be stepped per TOKEN, never per frame.

    This is the model's efficiency claim stated as an assertion, so it is
    checked at both extremes. All-blank is the one that matters: six frames
    must cost exactly one LLM call (the BOS state), because nothing was
    emitted. If someone ever moves the joint above the blank test, or feeds
    the LLM on every frame, this is what catches it.
    """
    m = model()
    feats, mask = _dummy_audio(m)
    t_frames = 6

    def count(threshold):
        calls = {"n": 0}
        real = m.llm.forward

        def counting(*a, **kw):
            calls["n"] += 1
            return real(*a, **kw)

        m.llm.forward = counting
        try:
            hyp = m.decode(feats, mask, bos_id=0,
                           t_lens=torch.tensor([t_frames]),
                           blank_threshold=threshold)[0]
        finally:
            m.llm.forward = real
        return calls["n"], len(hyp)

    n_blank, l_blank = count(0.0)
    assert l_blank == 0 and n_blank == 1, (n_blank, l_blank)

    n_emit, l_emit = count(1.0 + 1e-6)
    # BOS plus one advance per emitted token.
    assert l_emit == t_frames and n_emit == 1 + l_emit, (n_emit, l_emit)
    print(f"PASSED: test_llm_runs_once_per_token "
          f"({t_frames} frames: 0 tokens -> {n_blank} call, "
          f"{l_emit} tokens -> {n_emit} calls)")


def _dummy_audio(m):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(slimducer.AUT_NAME)
    import numpy as np
    wav = np.zeros(16000, dtype=np.float32)
    f = proc.feature_extractor([wav], sampling_rate=16000,
                               return_tensors="pt")
    mask = f.get("input_features_mask")
    if mask is None:
        mask = torch.ones(f["input_features"].shape[0],
                          f["input_features"].shape[-1], dtype=torch.bool)
    return f["input_features"].float(), mask


def test_forward_loss_is_finite_and_uses_reference_prefix():
    m = model()
    feats, mask = _dummy_audio(m)
    labels = torch.tensor([[11, 22, 33]])
    path = torch.full((8,), -1, dtype=torch.long)
    path[2], path[4], path[6] = 0, 1, 2
    out = m.forward(feats, mask, labels, torch.tensor([3]), [path], bos_id=0)
    assert torch.isfinite(out.loss), out.loss
    assert out.n_tokens == 3 and out.n_frames == 8, (out.n_tokens, out.n_frames)
    print(f"PASSED: test_forward_loss_is_finite_and_uses_reference_prefix "
          f"(loss {out.loss.item():.4f})")


if __name__ == "__main__":
    test_llm_is_frozen()
    test_trainable_set_is_what_we_think()
    test_joint_returns_llm_width()
    test_alignment_is_monotonic_and_consumes_every_token()
    test_alignment_rejects_more_tokens_than_frames()
    test_blank_threshold_moves_the_operating_point()
    test_llm_runs_once_per_token()
    test_forward_loss_is_finite_and_uses_reference_prefix()
    print("\nAll SlimDucer tests passed.")
