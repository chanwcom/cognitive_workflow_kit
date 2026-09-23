"""훈련 발화를 그대로 넣었을 때 제대로 받아쓰는지 확인한다.

dev WER 0.39 가 구조의 한계인지 일반화의 한계인지는 이 한 가지 측정으로
갈린다.  훈련 발화를 못 맞추면 구조나 용량이 문제고, 잘 맞추면 남은 문제는
순전히 일반화다.  같은 개수의 dev 발화를 같은 조건으로 함께 돌려 나란히
비교한다.
"""
from __future__ import annotations

import argparse

import torch

import slimducer
from slimducer import SlimDucer
from slimducer_train import DEV, TRAIN, batches, make_inputs, read_shards, wer


def evaluate(model, data, tok, fe, bos, device, max_frames, thresh, n_show,
             tag):
    err = ref = n_hyp = n_ref_tok = 0
    exact = 0
    shown = 0
    for idx in batches(data, max_frames):
        x, mask, labels, l_lens, t_lens = make_inputs(model, data, idx, fe,
                                                      device)
        with torch.no_grad():
            hyp = model.decode(x, mask, bos, torch.tensor(t_lens), thresh)
        for k, i in enumerate(idx):
            h = tok.decode(hyp[k])
            e, r = wer(data[i]["text"].split(), h.split())
            err += e
            ref += r
            n_hyp += len(hyp[k])
            n_ref_tok += len(data[i]["ids"])
            exact += int(e == 0)
            if shown < n_show:
                print(f"\n  [{tag} {shown}] T={t_lens[k]} "
                      f"참조토큰={len(data[i]['ids'])} 방출={len(hyp[k])} "
                      f"WER={e/max(r,1):.3f}")
                print(f"    REF: {data[i]['text'][:110]}")
                print(f"    HYP: {h[:110]!r}")
                shown += 1
    return {"wer": err / max(ref, 1), "exact": exact / max(len(data), 1),
            "emit_ratio": n_hyp / max(n_ref_tok, 1), "n": len(data)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--n_show", type=int, default=4)
    ap.add_argument("--max_sec", type=float, default=20.0)
    ap.add_argument("--max_frames", type=float, default=4800)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--branch_norm", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(slimducer.LLM_NAME)
    fe = AutoProcessor.from_pretrained(slimducer.AUT_NAME).feature_extractor
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    model = SlimDucer(dtype=torch.float32,
                      branch_norm=args.branch_norm).to(args.device).eval()
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model.joint.load_state_dict(ck["joint"])
    model.blank_head.load_state_dict(ck["blank_head"])
    model.audio_tower.load_state_dict(ck["audio_tower"])
    print(f"[ckpt] epoch {ck.get('epoch')}  branch_norm={args.branch_norm}",
          flush=True)

    train = read_shards(TRAIN, tok, args.max_sec, args.n_eval)
    dev = read_shards(DEV["dev-clean"], tok, args.max_sec, args.n_eval)
    print(f"[data] train {len(train)}  dev-clean {len(dev)}", flush=True)

    print("\n" + "=" * 74)
    print("훈련 발화")
    print("=" * 74)
    tr = evaluate(model, train, tok, fe, bos, args.device, args.max_frames,
                  args.thresh, args.n_show, "train")

    print("\n" + "=" * 74)
    print("dev-clean 발화 (같은 개수)")
    print("=" * 74)
    dv = evaluate(model, dev, tok, fe, bos, args.device, args.max_frames,
                  args.thresh, args.n_show, "dev")

    print("\n" + "=" * 74)
    print(f"{'split':>8} {'n':>5} {'WER':>8} {'완전일치':>9} {'방출/참조':>10}")
    print("=" * 74)
    for name, m in (("train", tr), ("dev-clean", dv)):
        print(f"{name:>8} {m['n']:>5} {m['wer']:>8.4f} {m['exact']:>9.3f} "
              f"{m['emit_ratio']:>10.2f}")


if __name__ == "__main__":
    main()
