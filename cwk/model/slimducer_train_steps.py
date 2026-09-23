"""Step-based SlimDucer fine-tuning on 100 h or 960 h LibriSpeech.

Three things differ from `slimducer_train.py`, which is epoch-based and
sized for the 10 h subset. All three are forced by the move to 100x more
data; none of them change the model.

**Alignment happens per batch, not per epoch.** The 10 h run realigns the
whole corpus once an epoch -- 30 passes over 2,762 utterances. The same
cadence on 960 h is 9.4 h per pass, and cutting it to four passes was
measured against the 10 h agreement trace and rejected: agreement there
needs eleven epochs to clear 97% (73 -> 86 -> 88 -> 90, a plateau, then
93 -> 97 -> 99), so a four-pass schedule stops inside the plateau. Aligning
each batch against the current model instead costs the same as one pass per
epoch -- the work is one alignment per utterance-visit either way -- and
leaves zero staleness. Measured 71.6 ms per utterance-visit against 86.3 for
the split version, because the frozen towers are encoded once and shared by
the alignment and the loss.

**TF32 is on.** The alignment scores a (T, L+1) grid through a 151,936-way
lm_head, which is 42x the fitting pass's T rows and about 85% of total run
time. TF32 cuts it 1.6x and reproduced fp32's path on 100% of utterances;
bf16 is 1.5x but changes 43% of paths, and fp16 changes 12%, so neither is
usable for a decision rule.

**Audio is held as int16.** 960 h of float32 is 221 GiB against 235 GiB
free, and a concurrent 100 h run needs its own. int16 halves it to 111 GiB
and costs one cast per batch.

The schedule follows the CTC and RNN-T convention already in use: a
400-second batch and warmup_stable_decay, 1000/10000/4000 over 15,000 steps
at 100 h and 1000/35000/14000 over 50,000 at 960 h. A 400-second batch does
not fit in 24 GiB -- 200 s peaks at 21.2 GiB and 400 s OOMs -- so it is
reached as 100 s x 4 gradient accumulation, which is also the fastest of the
measured points (980 s of audio per second, against 851 at 200 s: the tower
pads every utterance to 30 s, so larger batches buy nothing).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import tarfile
import time

import numpy as np
import torch
import torch.nn.functional as F

import slimducer
from slimducer import SlimDucer, SlimDucerOutput
from slimducer_train import wer

# The alignment is a decision rule over a 151,936-way softmax, so what
# matters is that TF32 leaves the argmax alone, which it did on every
# utterance measured. See the module docstring. It is on by default and
# --no_tf32 turns it off, because "TF32 does not change the alignment" is
# not the same claim as "TF32 does not change training".
def _set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


_set_tf32(True)

ROOT = "/mnt/synology_nas_00/datasets/asr"
DEV = {"dev-clean": f"{ROOT}/librispeech/webdataset/dev-clean",
       "dev-other": f"{ROOT}/librispeech/webdataset/dev-other"}
PRESETS = {
    # The 10 h subset the epoch-based script was tuned on. Kept here as the
    # control: it is the one setting whose answer is already known (0.336),
    # so running it through THIS script isolates the code and the batch from
    # the data.
    "10h": [f"{ROOT}/libri_light_finetuning/webdataset/10h"],
    "100h": [f"{ROOT}/librispeech/webdataset/train-clean-100"],
    "960h": [f"{ROOT}/librispeech/webdataset/train-clean-100",
             f"{ROOT}/librispeech/webdataset/train-clean-360",
             f"{ROOT}/librispeech/webdataset/train-other-500"],
}


def read_shards(patterns, tokenizer, max_sec, limit=None, tag=""):
    """Load utterances into RAM, audio kept as int16.

    `patterns` is a list of webdataset directories; 960 h is three of them.
    """
    import soundfile as sf
    out = []
    t0 = time.time()
    for pattern in patterns:
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
                    wav, sr = sf.read(io.BytesIO(raw["audio"]), dtype="int16")
                except Exception:
                    wav = np.frombuffer(raw["audio"], dtype=np.int16)
                    sr = 16000
                dur = len(wav) / sr
                if dur > max_sec:
                    continue
                out.append({"wav": np.ascontiguousarray(wav), "dur": dur,
                            "text": txt.lower(),
                            "ids": tokenizer.encode(txt.lower(),
                                                    add_special_tokens=False)})
                if limit and len(out) >= limit:
                    return out
            print(f"[read{tag}] {os.path.basename(pattern)}/"
                  f"{os.path.basename(shard)}: {len(out)} utt, "
                  f"{time.time()-t0:.0f}s", flush=True)
    return out


def bucket_batches(items, max_frames, max_utts, shuffle_rng=None):
    """Length-bucketed batches under TWO budgets, not one.

    A duration budget alone is not enough here. The feature extractor pads
    every utterance to 30 s and the tower runs on the padded tensor, so a
    batch's tower cost is set by its utterance COUNT, not by its audio. At a
    100 s budget LibriSpeech's short tail produces batches of up to 50
    utterances -- 98 s of speech but 1500 s once padded -- which measured
    20.2 GiB against 5.6 for a typical batch of six. Only 70 of 3889
    batches are that shape, but fifteen epochs draw each of them dozens of
    times, and one is enough to end the run.
    """
    order = sorted(range(len(items)), key=lambda i: items[i]["dur"])
    out, cur, cur_frames = [], [], 0.0
    for i in order:
        f = items[i]["dur"] * 100
        if cur and (cur_frames + f > max_frames or len(cur) >= max_utts):
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
    feats = fe([items[i]["wav"].astype(np.float32) / 32768.0 for i in idx],
               sampling_rate=16000, return_tensors="pt")
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


def loss_from_states(model, h_aut_list, h_llm_all, labels, label_lens,
                     alignment):
    """`SlimDucer.forward`'s objective against already-encoded towers.

    Identical arithmetic to `SlimDucer.forward`; it exists only so the
    encodings computed for the alignment are not thrown away and redone.
    `slimducer_train_steps_test.py` asserts the two agree bit for bit.
    """
    blank_terms, token_terms = [], []
    n_frames = n_tokens = 0
    for b, path in enumerate(alignment):
        t_len = int(path.shape[0])
        if t_len == 0 or bool((path < 0).all()) and int(label_lens[b]) > 0:
            continue
        emitted = (path >= 0).long()
        n_before = torch.cat([emitted.new_zeros(1), emitted.cumsum(0)[:-1]])
        h_llm = h_llm_all[b].index_select(0, n_before)
        h = model.joint(h_llm, h_aut_list[b][:t_len])

        logit_blank = model.blank_head(h).squeeze(-1).float()
        is_blank = (path < 0)
        blank_terms.append(F.binary_cross_entropy_with_logits(
            logit_blank, is_blank.to(logit_blank.dtype), reduction="sum"))

        emit_idx = (~is_blank).nonzero(as_tuple=True)[0]
        if emit_idx.numel():
            tgt = labels[b].index_select(0, path[emit_idx])
            logits = model.lm_head(h.index_select(0, emit_idx)).float()
            token_terms.append(F.cross_entropy(logits, tgt, reduction="sum"))
            n_tokens += int(emit_idx.numel())
        n_frames += t_len

    zero = h_aut_list[0].new_zeros((), dtype=torch.float32)
    blank_loss = torch.stack(blank_terms).sum() if blank_terms else zero
    token_loss = torch.stack(token_terms).sum() if token_terms else zero
    denom = max(n_frames, 1)
    return SlimDucerOutput(loss=(blank_loss + token_loss) / denom,
                           blank_loss=blank_loss / denom,
                           token_loss=token_loss / max(n_tokens, 1),
                           n_frames=n_frames, n_tokens=n_tokens)


def wsd_scale(step, warmup, stable, decay):
    """warmup_stable_decay, linear on both ramps, as in the CTC runs."""
    if step <= warmup:
        return step / max(warmup, 1)
    if step <= warmup + stable:
        return 1.0
    return max(0.0, 1.0 - (step - warmup - stable) / max(decay, 1))


def alignment_pass(model, train, fe, bos, device, max_frames, max_utts,
                   chunk, prev):
    """Align the whole corpus against a SNAPSHOT of the model.

    Aligning each batch against the live model looked free -- same cost, no
    staleness -- and it is wrong. Measured on the 10 h set against the
    epoch-based script's own 0.336: per-batch alignment reaches a training
    token loss of 0.42 while dev WER CLIMBS from 0.976 to 1.158, and the
    resulting model emits 0.31 tokens per reference token, i.e. it stops
    after one or two. The alignment and the weights co-adapt: the model
    aligns to what it already believes and is then trained to believe it
    harder, and nothing in the objective pulls it back. Freezing the targets
    for a whole pass is what makes this hard EM rather than a feedback loop.

    Returns the cached paths and the two diagnostics that catch a collapse:
    agreement with the previous pass, and emission spread against uniform.
    """
    model.eval()
    cache = {}
    t0 = time.time()
    with torch.no_grad():
        for idx in bucket_batches(train, max_frames, max_utts):
            x, mask, labels, l_lens, t_lens = make_inputs(
                model, train, idx, fe, device)
            h_aut = model.encode_audio(x, mask)
            h_llm = model.encode_text(labels, bos)
            for k, i in enumerate(idx):
                cache[i] = model.align(h_aut[k], h_llm[k], t_lens[k],
                                       int(l_lens[k]), labels[k],
                                       chunk=chunk).cpu()
    model.train()

    ok = [i for i in cache if bool((cache[i] >= 0).any())]
    spread, agree = [], []
    for i in ok:
        pos = (cache[i] >= 0).nonzero().flatten().float()
        if pos.numel() > 1:
            spread.append(float(pos.std() / len(cache[i])))
        if i in prev and prev[i].shape == cache[i].shape:
            agree.append(float((prev[i] == cache[i]).float().mean()))
    return cache, {"usable": len(ok), "total": len(cache),
                   "secs": time.time() - t0,
                   "spread": float(np.mean(spread)) if spread else float("nan"),
                   "agree": float(np.mean(agree)) if agree else float("nan")}


def align_window(model, train, window, fe, bos, device, chunk, last):
    """Align only the utterances the next window of steps will consume.

    A full-corpus pass every `realign_every` steps costs
    (steps_per_epoch / realign_every) times more than one pass per epoch,
    because it realigns utterances that have not been used since the last
    time. Aligning just the upcoming window costs the same as one pass per
    epoch -- each utterance is still aligned once per epoch -- while
    bounding how stale a target can be by the window rather than by the
    epoch. That distinction matters as the corpus grows: one epoch is 833
    steps at 10 h but 72,000 at 960 h, and the model is not the same model
    72,000 steps later.
    """
    model.eval()
    cache = {}
    t0 = time.time()
    with torch.no_grad():
        for idx in window:
            x, mask, labels, l_lens, t_lens = make_inputs(
                model, train, idx, fe, device)
            h_aut = model.encode_audio(x, mask)
            h_llm = model.encode_text(labels, bos)
            for k, i in enumerate(idx):
                cache[i] = model.align(h_aut[k], h_llm[k], t_lens[k],
                                       int(l_lens[k]), labels[k],
                                       chunk=chunk).cpu()
    model.train()

    ok = [i for i in cache if bool((cache[i] >= 0).any())]
    spread, agree = [], []
    for i in ok:
        pos = (cache[i] >= 0).nonzero().flatten().float()
        if pos.numel() > 1:
            spread.append(float(pos.std() / len(cache[i])))
        if i in last and last[i].shape == cache[i].shape:
            agree.append(float((last[i] == cache[i]).float().mean()))
    last.update(cache)
    return cache, {"usable": len(ok), "total": len(cache),
                   "secs": time.time() - t0,
                   "spread": float(np.mean(spread)) if spread else float("nan"),
                   "agree": float(np.mean(agree)) if agree else float("nan")}


def uniform_alignment(train, model, fe, device, max_frames, max_utts):
    """Place the L tokens evenly across the T frames, ignoring the model.

    Round zero is the one alignment computed from weights that have learned
    nothing, and "random" is the wrong word for what it produces: Viterbi
    maximises whatever the initialised joint happens to prefer, so the path
    is systematically biased rather than uninformative, and the next epoch
    is trained to confirm that bias. Measured, the same configuration has
    landed at 0.276, 0.336 and 0.96 dev WER, and the failures are the runs
    whose emission spread SHRANK from 0.266 toward 0.253 instead of growing
    toward the uniform 0.289 that the successful runs settle on.

    Even spacing starts at that settling point. It ignores the acoustics,
    which is a real cost on utterances with uneven speaking rate, but it
    costs only the first round: from epoch two the model is aligning with
    weights that have seen the data.
    """
    align = {}
    for idx in bucket_batches(train, max_frames, max_utts):
        for i in idx:
            t_len = model.audio_out_len(int(round(train[i]["dur"] * 100)))
            l_len = len(train[i]["ids"])
            path = torch.full((t_len,), -1, dtype=torch.long)
            if 0 < l_len <= t_len:
                # token l sits at the centre of the l-th of L equal slices
                pos = [min(t_len - 1, int((l + 0.5) * t_len / l_len))
                       for l in range(l_len)]
                # keep them strictly increasing, as an alignment must be
                for l in range(1, l_len):
                    if pos[l] <= pos[l - 1]:
                        pos[l] = pos[l - 1] + 1
                if pos[-1] < t_len:
                    for l, t in enumerate(pos):
                        path[t] = l
            align[i] = path
    spread = []
    for p in align.values():
        q = (p >= 0).nonzero().flatten().float()
        if q.numel() > 1:
            spread.append(float(q.std() / len(p)))
    return align, {"usable": sum(1 for p in align.values()
                                 if bool((p >= 0).any())),
                   "total": len(align), "secs": 0.0,
                   "spread": float(np.mean(spread)) if spread else float("nan"),
                   "agree": float("nan")}


def evaluate(model, devs, tok, fe, bos, device, max_frames, max_utts,
             thresh):
    model.eval()
    out = {}
    for name, data in devs.items():
        err = ref = 0
        for idx in bucket_batches(data, max_frames, max_utts):
            x, mask, labels, l_lens, t_lens = make_inputs(
                model, data, idx, fe, device)
            with torch.no_grad():
                hyp = model.decode(x, mask, bos, torch.tensor(t_lens), thresh)
            for k, i in enumerate(idx):
                e, r = wer(data[i]["text"].split(), tok.decode(hyp[k]).split())
                err += e
                ref += r
        out[f"{name}_wer"] = err / max(ref, 1)
    model.train()
    return out


def save(model, path, step):
    torch.save({"joint": model.joint.state_dict(),
                "blank_head": model.blank_head.state_dict(),
                "audio_tower": model.audio_tower.state_dict(),
                "step": step}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), required=True)
    ap.add_argument("--max_steps", type=int, required=True)
    ap.add_argument("--warmup_steps", type=int, required=True)
    ap.add_argument("--num_stable_steps", type=int, required=True)
    ap.add_argument("--num_decay_steps", type=int, required=True)
    ap.add_argument("--max_frames", type=float, default=10000,
                    help="mel frames per MICRO-batch (100 per second). 10000 "
                         "is 100 s, the fastest point measured that fits: "
                         "200 s peaks at 21.2 GiB and 400 s OOMs on 24 GiB.")
    ap.add_argument("--max_batch_utts", type=int, default=16,
                    help="utterances per micro-batch. The tower pads every "
                         "utterance to 30 s, so this and not the duration "
                         "budget bounds memory: B=50 measured 20.2 GiB, "
                         "B=16 about 8.7.")
    ap.add_argument("--grad_accum", type=int, default=4,
                    help="micro-batches per optimizer step. 100 s x 4 is the "
                         "400 s batch the CTC and RNN-T runs use.")
    ap.add_argument("--max_sec", type=float, default=20.0)
    ap.add_argument("--lr_joint", type=float, default=5e-4)
    ap.add_argument("--lr_audio", type=float, default=2e-5)
    ap.add_argument("--realign_every", type=int, default=0,
                    help="steps between full alignment passes. 0 aligns each "
                         "batch against the live model, which measured a "
                         "collapse (see alignment_pass); pass the number of "
                         "steps in one epoch to match the epoch-based "
                         "script.")
    ap.add_argument("--init_align", choices=["model", "uniform"],
                    default="model",
                    help="what round zero aligns against: the initialised "
                         "model (default, as the epoch-based script did) or "
                         "even spacing. See uniform_alignment.")
    ap.add_argument("--no_tf32", action="store_true",
                    help="run the matmuls in full fp32, as the epoch-based "
                         "script did.")
    ap.add_argument("--exact_forward", action="store_true",
                    help="route the loss through SlimDucer.forward instead "
                         "of the fused loss_from_states, re-encoding the "
                         "frozen towers. Slower by about 17%%; exists so a "
                         "reproduction can remove that difference too.")
    ap.add_argument("--rolling_align", action="store_true",
                    help="align only the upcoming --realign_every steps' "
                         "worth of utterances instead of the whole corpus. "
                         "Same cost per epoch, staleness bounded by the "
                         "window rather than by the epoch.")
    ap.add_argument("--align_chunk", type=int, default=8,
                    help="frames per alignment chunk. This, not the "
                         "batch, sets peak memory: the chunk holds a "
                         "(chunk, L+1, 151936) logit tensor, so the "
                         "worst batch peaks at 5.8 GiB here against "
                         "16.9 at 32, and is no slower on long "
                         "utterances.")
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_steps", default="",
                    help="comma-separated steps to keep a named checkpoint at")
    ap.add_argument("--dev_limit", type=int, default=200)
    ap.add_argument("--blank_threshold", type=float, default=0.5)
    ap.add_argument("--freeze_audio", action="store_true")
    ap.add_argument("--branch_norm", action="store_true")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    _set_tf32(not args.no_tf32)
    total = args.warmup_steps + args.num_stable_steps + args.num_decay_steps
    assert total == args.max_steps, (
        f"warmup+stable+decay = {total} but max_steps = {args.max_steps}; "
        "the three must sum, as in the CTC profiles")
    keep = [int(s) for s in args.save_steps.split(",") if s.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from transformers import AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(slimducer.LLM_NAME)
    fe = AutoProcessor.from_pretrained(slimducer.AUT_NAME).feature_extractor
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    train = read_shards(PRESETS[args.preset], tok, args.max_sec)
    devs = {k: read_shards([v], tok, args.max_sec, args.dev_limit, tag=" dev")
            for k, v in DEV.items()}
    hrs = sum(u["dur"] for u in train) / 3600
    gib = sum(u["wav"].nbytes for u in train) / 2 ** 30
    print(f"[data] train {len(train)} utt ({hrs:.1f} h, {gib:.0f} GiB int16), "
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

    per_epoch = len(bucket_batches(train, args.max_frames,
                                   args.max_batch_utts))
    steps_per_epoch = per_epoch / args.grad_accum
    print(f"[plan] {per_epoch} micro-batches/epoch, "
          f"{steps_per_epoch:.0f} steps/epoch, "
          f"{args.max_steps/steps_per_epoch:.1f} epochs over {args.max_steps} "
          f"steps; batch {args.max_frames/100:.0f}s x {args.grad_accum} = "
          f"{args.max_frames*args.grad_accum/100:.0f}s", flush=True)

    def micro_batches():
        while True:
            for b in bucket_batches(train, args.max_frames,
                                    args.max_batch_utts, rng):
                yield b

    stream = micro_batches()
    model.train()
    t0 = time.time()
    run = {"loss": 0.0, "blank": 0.0, "token": 0.0, "n": 0}
    logf = open(os.path.join(args.out_dir, "log.jsonl"), "a")

    align_cache: dict[int, torch.Tensor] = {}
    prev_align: dict[int, torch.Tensor] = {}
    pending: list[list[int]] = []

    for step in range(1, args.max_steps + 1):
        if args.realign_every and (step - 1) % args.realign_every == 0:
            if step == 1 and args.init_align == "uniform":
                align_cache, st = uniform_alignment(
                    train, model, fe, args.device, args.max_frames,
                    args.max_batch_utts)
                pending = []
            elif args.rolling_align:
                window = [next(stream) for _ in
                          range(args.realign_every * args.grad_accum)]
                align_cache, st = align_window(
                    model, train, window, fe, bos, args.device,
                    args.align_chunk, prev_align)
                pending = list(window)
            else:
                prev_align = align_cache
                align_cache, st = alignment_pass(
                    model, train, fe, bos, args.device, args.max_frames,
                    args.max_batch_utts, args.align_chunk, prev_align)
            print(f"[align step{step}] {st['usable']}/{st['total']} usable, "
                  f"{st['secs']:.0f}s, spread {st['spread']:.3f} "
                  f"(uniform~0.289), agree {st['agree']*100:.1f}%", flush=True)

        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            idx = (pending.pop(0) if (args.rolling_align and pending)
                   else next(stream))
            x, mask, labels, l_lens, t_lens = make_inputs(
                model, train, idx, fe, args.device)
            # The towers are frozen, so one encoding serves both the
            # alignment and the loss. If the tower is being trained the
            # loss needs its own graph, so it is re-encoded there.
            with torch.no_grad():
                h_aut = model.encode_audio(x, mask)
                h_llm = model.encode_text(labels, bos)
                if args.realign_every:
                    paths = [align_cache[i].to(args.device) for i in idx]
                else:
                    paths = [model.align(h_aut[k], h_llm[k], t_lens[k],
                                         int(l_lens[k]), labels[k],
                                         chunk=args.align_chunk)
                             for k in range(len(idx))]
            if not any(bool((p >= 0).any()) for p in paths):
                continue
            if aud or args.exact_forward:
                out = model(x, mask, labels, l_lens, paths, bos)
            else:
                out = loss_from_states(model, h_aut, h_llm, labels, l_lens,
                                       paths)
            (out.loss / args.grad_accum).backward()
            run["loss"] += out.loss.item()
            run["blank"] += out.blank_loss.item()
            run["token"] += out.token_loss.item()
            run["n"] += 1

        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        scale = wsd_scale(step, args.warmup_steps, args.num_stable_steps,
                          args.num_decay_steps)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * scale
        opt.step()

        if step % args.log_every == 0:
            n = max(run["n"], 1)
            print(f"[{step}/{args.max_steps}] loss={run['loss']/n:.4f} "
                  f"blank={run['blank']/n:.4f} token={run['token']/n:.4f} "
                  f"lr={base[0]*scale:.2e} "
                  f"alloc={torch.cuda.max_memory_allocated()/2**30:.1f} "
                  f"resv={torch.cuda.max_memory_reserved()/2**30:.1f}GiB "
                  f"{(time.time()-t0)/step:.2f}s/step "
                  f"eta={(args.max_steps-step)*(time.time()-t0)/step/3600:.1f}h",
                  flush=True)
            run = {"loss": 0.0, "blank": 0.0, "token": 0.0, "n": 0}

        if step % args.eval_steps == 0 or step == args.max_steps:
            rep = {"step": step, "lr": base[0] * scale,
                   "wall_min": (time.time() - t0) / 60,
                   "peak_gib": torch.cuda.max_memory_reserved() / 2 ** 30}
            rep.update(evaluate(model, devs, tok, fe, bos, args.device,
                                args.max_frames, args.max_batch_utts,
                                args.blank_threshold))
            print("[eval] " + json.dumps(rep), flush=True)
            logf.write(json.dumps(rep) + "\n")
            logf.flush()
            save(model, os.path.join(args.out_dir, "ckpt.pt"), step)

        if step in keep:
            save(model, os.path.join(args.out_dir, f"ckpt_step{step}.pt"), step)
            print(f"[save] step {step}", flush=True)

    logf.close()
    print(f"[done] {args.max_steps} steps in "
          f"{(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
