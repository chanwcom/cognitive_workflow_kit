"""Look at what the trained SlimDucer actually emits, and sweep the blank threshold.

Training loss falls while dev WER sits above 1.0, which means insertions
outnumber the reference. That shape has two very different causes and the
fix differs, so read the hypotheses before changing anything:

  * a repeat loop -- the emission decision is stuck open, and the threshold
    is the knob for it;
  * output unrelated to the reference -- the conditioning has come apart,
    and no threshold will save it.

The threshold sweep is free in the sense that it needs no retraining: the
emission decision is a single sigmoid, so the operating point moves without
touching a weight.
"""
from __future__ import annotations

import argparse
import json

import torch

import slimducer
from slimducer import SlimDucer
from slimducer_train import DEV, batches, make_inputs, read_shards, wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--n_show", type=int, default=6)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--max_sec", type=float, default=20.0)
    ap.add_argument("--max_frames", type=float, default=4800)
    ap.add_argument("--thresholds", default="0.3,0.5,0.7,0.9,0.95,0.99")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(slimducer.LLM_NAME)
    fe = AutoProcessor.from_pretrained(slimducer.AUT_NAME).feature_extractor
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    model = SlimDucer(dtype=torch.float32).to(args.device).eval()
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model.joint.load_state_dict(ck["joint"])
    model.blank_head.load_state_dict(ck["blank_head"])
    model.audio_tower.load_state_dict(ck["audio_tower"])
    print(f"[ckpt] epoch {ck.get('epoch')}", flush=True)

    data = read_shards(DEV[args.split], tok, args.max_sec, args.n_eval)
    print(f"[data] {args.split}: {len(data)} utterances", flush=True)

    print("\n" + "=" * 78)
    print(f"샘플 {args.n_show}개 (threshold 0.5)")
    print("=" * 78)
    shown = 0
    for idx in batches(data, args.max_frames):
        x, mask, labels, l_lens, t_lens = make_inputs(model, data, idx, fe,
                                                      args.device)
        with torch.no_grad():
            hyp = model.decode(x, mask, bos, torch.tensor(t_lens), 0.5)
        for k, i in enumerate(idx):
            if shown >= args.n_show:
                break
            h = tok.decode(hyp[k])
            e, r = wer(data[i]["text"].split(), h.split())
            print(f"\n[{shown}] T={t_lens[k]} 참조토큰={len(data[i]['ids'])} "
                  f"방출={len(hyp[k])}  WER={e/max(r,1):.3f}")
            print(f"  REF: {data[i]['text'][:120]}")
            print(f"  HYP: {h[:120]!r}")
            shown += 1
        if shown >= args.n_show:
            break

    print("\n" + "=" * 78)
    print("blank threshold 스윕")
    print("=" * 78)
    print(f"{'thresh':>7} {'WER':>8} {'sub':>7} {'del':>7} {'ins':>7} "
          f"{'방출/참조':>10}")
    for th in [float(t) for t in args.thresholds.split(",")]:
        err = ref = n_hyp = n_ref_tok = 0
        for idx in batches(data, args.max_frames):
            x, mask, labels, l_lens, t_lens = make_inputs(model, data, idx,
                                                          fe, args.device)
            with torch.no_grad():
                hyp = model.decode(x, mask, bos, torch.tensor(t_lens), th)
            for k, i in enumerate(idx):
                e, r = wer(data[i]["text"].split(),
                           tok.decode(hyp[k]).split())
                err += e
                ref += r
                n_hyp += len(hyp[k])
                n_ref_tok += len(data[i]["ids"])
        print(f"{th:>7.2f} {err/max(ref,1):>8.4f} {'-':>7} {'-':>7} {'-':>7} "
              f"{n_hyp/max(n_ref_tok,1):>10.2f}")


if __name__ == "__main__":
    main()
