"""Stage-5: fit SlimDucer on the 10 h Libri-Light subset.

Viterbi training. Each epoch realigns the whole set against the current
model, then does teacher forcing against those fixed frame targets.

Why the alignment pass is affordable at all: the emission score at frame t
for token l depends on the LLM state after l-1 tokens, so the grid is
(T, L+1) rather than (T,) -- about 35 k positions for a 30 s utterance
against a 90-token transcript. That is 5.5 TFLOP of lm_head per utterance,
a tenth of a second, so a full pass over 2,763 utterances costs minutes,
not hours. Memory is the binding constraint, not compute, which is why the
grid is chunked over t.

Logged every epoch, because the prompt for this work asks for them and
because each one catches a different failure:

  * train vs dev loss -- 10 h against a 185 M trainable encoder is the
    regime where this project has already seen a model reach a LOWER
    training loss than a 10x larger run while doubling WER.
  * frame agreement with the previous alignment -- hard EM has no
    guarantee of settling; a figure that keeps swinging means it has not.
  * emission spread against uniform -- the giveaway for a degenerate
    alignment that has crammed the transcript into one end.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import tarfile
import time

import numpy as np
import torch

import slimducer
from slimducer import SlimDucer

ROOT = "/mnt/synology_nas_00/datasets/asr"
TRAIN = f"{ROOT}/libri_light_finetuning/webdataset/10h"
DEV = {"dev-clean": f"{ROOT}/librispeech/webdataset/dev-clean",
       "dev-other": f"{ROOT}/librispeech/webdataset/dev-other"}


def read_shards(pattern, tokenizer, max_sec, limit=None):
    import soundfile as sf
    out = []
    for shard in sorted(glob.glob(pattern + "/*.tar")):
        t = tarfile.open(shard)
        cur = {}
        for m in t:
            key, ext = m.name.rsplit(".", 1)
            if ext not in ("audio", "text"):
                continue
            cur.setdefault(key, {})[ext] = t.extractfile(m).read()
            if len(cur[key]) < 2:
                continue
            raw = cur.pop(key)
            txt = raw["text"].decode().strip()
            try:
                wav, sr = sf.read(io.BytesIO(raw["audio"]), dtype="float32")
            except Exception:
                wav = np.frombuffer(raw["audio"], dtype=np.int16)
                wav, sr = wav.astype(np.float32) / 32768.0, 16000
            dur = len(wav) / sr
            if dur > max_sec:
                continue
            ids = tokenizer.encode(txt.lower(), add_special_tokens=False)
            out.append({"wav": wav, "dur": dur, "text": txt.lower(),
                        "ids": ids})
            if limit and len(out) >= limit:
                return out
    return out


def wer(ref_words, hyp_words):
    """Plain Levenshtein at the word level.

    Written out rather than pulled from `evaluate`: that library writes an
    arrow cache under the home directory, and a full root filesystem there
    has already killed a 13,000-step run on this machine.
    """
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(ref_words) + 1)
    d[0, :] = np.arange(len(hyp_words) + 1)
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + cost)
    return int(d[-1, -1]), len(ref_words)


def batches(items, max_frames, shuffle_rng=None):
    """Length-bucketed batches.

    Sorting by duration first keeps a 2 s utterance out of the same padded
    tensor as a 28 s one; the feature extractor pads everything to 30 s
    anyway, so the saving is in the joint and the alignment, not the mel.
    """
    order = sorted(range(len(items)), key=lambda i: items[i]["dur"])
    out, cur, cur_frames = [], [], 0.0
    for i in order:
        f = items[i]["dur"] * SlimDucer.MEL_HZ if hasattr(SlimDucer, "MEL_HZ") \
            else items[i]["dur"] * 100
        if cur and cur_frames + f > max_frames:
            out.append(cur)
            cur, cur_frames = [], 0.0
        cur.append(i)
        cur_frames += f
    if cur:
        out.append(cur)
    if shuffle_rng is not None:
        shuffle_rng.shuffle(out)
    return out


def make_inputs(model, items, idx, fe, device):
    feats = fe([items[i]["wav"] for i in idx], sampling_rate=16000,
               return_tensors="pt")
    x = feats["input_features"].to(device)
    mask = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.bool)
    for k, i in enumerate(idx):
        mask[k, :int(round(items[i]["dur"] * 100))] = True
    mask = mask.to(device)
    t_lens = [model.audio_out_len(int(m.sum())) for m in mask]
    l_max = max(len(items[i]["ids"]) for i in idx)
    labels = torch.zeros((len(idx), max(l_max, 1)), dtype=torch.long,
                         device=device)
    for k, i in enumerate(idx):
        ids = items[i]["ids"]
        labels[k, :len(ids)] = torch.tensor(ids, device=device)
    l_lens = torch.tensor([len(items[i]["ids"]) for i in idx], device=device)
    return x, mask, labels, l_lens, t_lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max_sec", type=float, default=20.0)
    ap.add_argument("--max_frames", type=float, default=1600,
                    help="mel frames per batch (100 per second)")
    ap.add_argument("--lr_joint", type=float, default=5e-4)
    ap.add_argument("--lr_audio", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--align_chunk", type=int, default=8)
    ap.add_argument("--realign_every", type=int, default=1, help="epochs")
    ap.add_argument("--dev_limit", type=int, default=300)
    ap.add_argument("--blank_threshold", type=float, default=0.5)
    ap.add_argument("--freeze_audio", action="store_true",
                    help="Train only the joint and the blank head. AuT was "
                         "pre-trained on 20 M hours; re-fitting its 185 M "
                         "weights on 9.7 h of labels measured 22 %% dev "
                         "frame accuracy against a training loss of 0.30, "
                         "i.e. it memorised. Freezing leaves 3 M trainable.")
    ap.add_argument("--branch_norm", action="store_true",
                    help="LayerNorm each joint input before summing. "
                         "Measured without it: h_llm enters 3.15x larger "
                         "than h_aut, and the ablation gives the LLM twice "
                         "the audio's share of frame accuracy.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from transformers import AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(slimducer.LLM_NAME)
    fe = AutoProcessor.from_pretrained(slimducer.AUT_NAME).feature_extractor
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    train = read_shards(TRAIN, tok, args.max_sec)
    devs = {k: read_shards(v, tok, args.max_sec, args.dev_limit)
            for k, v in DEV.items()}
    print(f"[data] train {len(train)} utt "
          f"({sum(u['dur'] for u in train)/3600:.2f} h), "
          + ", ".join(f"{k} {len(v)}" for k, v in devs.items()), flush=True)

    model = SlimDucer(dtype=torch.float32,
                      freeze_audio=args.freeze_audio,
                      branch_norm=args.branch_norm).to(args.device)
    n_tr = sum(p.numel() for p in model.trainable_parameters())
    print(f"[model] trainable {n_tr/1e6:.1f}M  frozen "
          f"{(sum(p.numel() for p in model.parameters()) - n_tr)/1e6:.1f}M",
          flush=True)

    head = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("audio_tower.")]
    aud = [p for n, p in model.named_parameters()
           if p.requires_grad and n.startswith("audio_tower.")]
    groups = [{"params": head, "lr": args.lr_joint}]
    base = [args.lr_joint]
    if aud:
        groups.append({"params": aud, "lr": args.lr_audio})
        base.append(args.lr_audio)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    bat = batches(train, args.max_frames)
    total = args.epochs * len(bat)
    print(f"[plan] {len(bat)} batches/epoch, {total} steps", flush=True)

    align: dict[int, torch.Tensor] = {}
    prev_align: dict[int, torch.Tensor] = {}
    step = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        if (ep - 1) % args.realign_every == 0:
            model.eval()
            prev_align, align = align, {}
            ta = time.time()
            with torch.no_grad():
                for idx in bat:
                    x, mask, labels, l_lens, t_lens = make_inputs(
                        model, train, idx, fe, args.device)
                    h_aut = model.encode_audio(x, mask)
                    h_llm = model.encode_text(labels, bos)
                    for k, i in enumerate(idx):
                        align[i] = model.align(h_aut[k], h_llm[k], t_lens[k],
                                               int(l_lens[k]), labels[k],
                                               chunk=args.align_chunk).cpu()
            ok = [i for i in align if bool((align[i] >= 0).any())]
            spread, agree = [], []
            for i in ok:
                pos = (align[i] >= 0).nonzero().flatten().float()
                if pos.numel() > 1:
                    spread.append(float(pos.std() / len(align[i])))
                if i in prev_align and prev_align[i].shape == align[i].shape:
                    agree.append(float((prev_align[i] == align[i]).float().mean()))
            print(f"[align ep{ep}] {len(ok)}/{len(align)} usable, "
                  f"{time.time()-ta:.0f}s, 방출위치 표준편차/T "
                  f"{np.mean(spread):.3f} (균등≈0.289)"
                  + (f", 직전과 일치율 {np.mean(agree)*100:.1f}%" if agree else ""),
                  flush=True)
            model.train()

        model.train()
        run_loss = run_blank = run_token = 0.0
        n_b = 0
        for idx in batches(train, args.max_frames, rng):
            paths = [align[i] for i in idx]
            if not any(bool((p >= 0).any()) for p in paths):
                continue
            x, mask, labels, l_lens, t_lens = make_inputs(
                model, train, idx, fe, args.device)
            out = model(x, mask, labels, l_lens,
                        [p.to(args.device) for p in paths], bos)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            step += 1
            scale = min(1.0, step / max(1, args.warmup))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * scale
            opt.step()
            run_loss += out.loss.item(); run_blank += out.blank_loss.item()
            run_token += out.token_loss.item(); n_b += 1

        model.eval()
        report = {"epoch": ep, "step": step,
                  "train_loss": run_loss / max(n_b, 1),
                  "train_blank": run_blank / max(n_b, 1),
                  "train_token": run_token / max(n_b, 1),
                  "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                  "wall_min": (time.time() - t0) / 60}
        for name, data in devs.items():
            err = ref = 0
            for idx in batches(data, args.max_frames):
                x, mask, labels, l_lens, t_lens = make_inputs(
                    model, data, idx, fe, args.device)
                with torch.no_grad():
                    hyp = model.decode(x, mask, bos,
                                       torch.tensor(t_lens),
                                       args.blank_threshold)
                for k, i in enumerate(idx):
                    e, r = wer(data[i]["text"].split(),
                               tok.decode(hyp[k]).split())
                    err += e; ref += r
            report[f"{name}_wer"] = err / max(ref, 1)
        print("[epoch] " + json.dumps(report), flush=True)
        with open(os.path.join(args.out_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(report) + "\n")
        torch.save({"joint": model.joint.state_dict(),
                    "blank_head": model.blank_head.state_dict(),
                    "audio_tower": model.audio_tower.state_dict(),
                    "epoch": ep},
                   os.path.join(args.out_dir, "ckpt.pt"))


if __name__ == "__main__":
    main()
