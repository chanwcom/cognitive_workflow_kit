"""Stage-4 gate: can SlimDucer fit a handful of utterances at all?

The open question is not accuracy, it is whether a 3 M-parameter joint can
steer a FROZEN 152 k-wide output projection onto the right token. If a
couple of utterances cannot be memorised, nothing is gained by spending a
day on 10 h, so this runs first and reports.

What a pass looks like: the token term falls from ~ln(152k) = 11.9 toward
zero while the blank term also falls. A blank term that collapses while the
token term sits at 11.9 means the model learned "say nothing" and the
alignment is carrying the loss -- that is a fail, not a pass.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import tarfile
import time

import numpy as np
import torch

import slimducer
from slimducer import SlimDucer

DATA = ("/mnt/synology_nas_00/datasets/asr/"
        "libri_light_finetuning/webdataset/10h")
# 30 s of audio become 390 AuT frames, so the encoder runs at 13 Hz. The
# feature extractor pads everything to 30 s, which is why the valid length
# has to be recovered from the sample count rather than the tensor shape.
FRAMES_PER_SEC = 390.0 / 30.0


def load_utterances(n, max_sec, processor, tokenizer):
    import soundfile as sf
    out = []
    for shard in sorted(glob.glob(DATA + "/*.tar")):
        t = tarfile.open(shard)
        cur = {}
        for m in t:
            key, ext = m.name.rsplit(".", 1)
            if ext in ("audio", "text"):
                cur.setdefault(key, {})[ext] = t.extractfile(m).read()
            if len(cur.get(key, {})) < 2:
                continue
            raw, txt = cur[key]["audio"], cur[key]["text"].decode().strip()
            try:
                wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
            except Exception:
                wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                wav, sr = wav / 32768.0, 16000
            dur = len(wav) / sr
            if dur > max_sec:
                continue
            ids = tokenizer.encode(txt.lower(), add_special_tokens=False)
            if len(ids) >= dur * FRAMES_PER_SEC:      # cannot possibly fit
                continue
            out.append((wav, dur, txt, ids))
            if len(out) >= n:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_utt", type=int, default=2)
    ap.add_argument("--max_sec", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--realign_every", type=int, default=50)
    ap.add_argument("--lr_joint", type=float, default=1e-3)
    ap.add_argument("--lr_audio", type=float, default=1e-5)
    ap.add_argument("--freeze_audio", action="store_true")
    ap.add_argument("--align_chunk", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(slimducer.LLM_NAME)
    processor = AutoProcessor.from_pretrained(slimducer.AUT_NAME)
    bos = tokenizer.bos_token_id
    if bos is None:                       # Qwen3 has no dedicated BOS
        bos = tokenizer.eos_token_id
    print(f"[cfg] bos_id={bos}", flush=True)

    utts = load_utterances(args.n_utt, args.max_sec, processor, tokenizer)
    print(f"[data] {len(utts)} utterances", flush=True)
    for wav, dur, txt, ids in utts:
        print(f"   {dur:5.2f}s  T={int(dur*FRAMES_PER_SEC):3d}  L={len(ids):3d}"
              f"  T/L={dur*FRAMES_PER_SEC/len(ids):4.2f}  {txt[:52]!r}",
              flush=True)

    model = SlimDucer(freeze_audio=args.freeze_audio,
                      dtype=torch.float32).to(args.device)
    model.train()
    n_tr = sum(p.numel() for p in model.trainable_parameters())
    print(f"[model] trainable {n_tr/1e6:.1f}M", flush=True)

    feats = processor.feature_extractor(
        [u[0] for u in utts], sampling_rate=16000, return_tensors="pt")
    x = feats["input_features"].to(args.device)
    # 실제 길이 마스크. 특징 추출기가 30초로 패딩하므로 이걸 안 주면
    # 모든 발화가 390프레임이 되고 정렬이 무음에 토큰을 배정한다.
    fmask = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.bool)
    for i, u in enumerate(utts):
        fmask[i, :int(round(u[1] * 100))] = True      # mel hop 10 ms
    fmask = fmask.to(args.device)
    t_lens = [model.audio_out_len(int(m.sum())) for m in fmask]
    l_max = max(len(u[3]) for u in utts)
    labels = torch.zeros((len(utts), l_max), dtype=torch.long,
                         device=args.device)
    for i, u in enumerate(utts):
        labels[i, :len(u[3])] = torch.tensor(u[3], device=args.device)
    label_lens = torch.tensor([len(u[3]) for u in utts], device=args.device)

    head = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("audio_tower.")]
    aud = [p for n, p in model.named_parameters()
           if p.requires_grad and n.startswith("audio_tower.")]
    # Separate groups because the audio tower is pre-trained and the joint
    # is random; one rate either wrecks the encoder or starves the joint.
    groups = [{"params": head, "lr": args.lr_joint}]
    if aud:
        groups.append({"params": aud, "lr": args.lr_audio})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

    align = None
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if align is None or step % args.realign_every == 1:
            model.eval()
            with torch.no_grad():
                h_aut = model.encode_audio(x, fmask)
                h_llm = model.encode_text(labels, bos)
                new = [model.align(h_aut[i][:t_lens[i]], h_llm[i],
                                   t_lens[i], int(label_lens[i]),
                                   labels[i], chunk=args.align_chunk)
                       for i in range(len(utts))]
            model.train()
            if align is not None:
                same = np.mean([float((a == b).float().mean())
                                for a, b in zip(align, new)])
                print(f"[align] step {step}: 직전과 프레임 일치율 {same*100:5.1f}%",
                      flush=True)
            align = new

        out = model(x, fmask, labels, label_lens, align, bos)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()

        if step % 20 == 0 or step == 1:
            mem = torch.cuda.max_memory_allocated() / 2 ** 30
            print(f"[{step:4d}/{args.steps}] loss={out.loss.item():8.4f} "
                  f"blank={out.blank_loss.item():7.4f} "
                  f"token={out.token_loss.item():8.4f} "
                  f"(random={np.log(model.vocab_size):.2f}) "
                  f"peak={mem:5.2f}GiB {(time.time()-t0)/step:5.2f}s/step",
                  flush=True)

    model.eval()
    hyp = model.decode(x, fmask, bos, torch.tensor(t_lens), 0.5)
    for i, u in enumerate(utts):
        print(f"\n참조 : {u[2].lower()[:90]}")
        print(f"가설 : {tokenizer.decode(hyp[i])[:90]!r}")


if __name__ == "__main__":
    main()
