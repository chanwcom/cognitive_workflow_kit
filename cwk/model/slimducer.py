"""SlimDucer: speech recognition by injecting acoustics beneath a frozen LLM's softmax.

The usual way to make an LLM transcribe is to project encoder features into
its INPUT embedding space and prepend them. The audio then occupies context
and traverses every transformer layer, and competitive accuracy needs LoRA
or full fine-tuning, which costs the backbone its general ability.

SlimDucer couples at the other end of the stack. A trainable joint MLP maps
the audio encoder's penultimate representation and the frozen LLM's final
hidden state into the LLM's own hidden space, where the LLM's ORIGINAL
output projection turns it into a distribution over tokens. Acoustics steer
next-token prediction directly and never enter the LLM's context.

    audio -> AuT (trained) -> h_aut(t) -----.
                                             +-> Joint -> h(t) -+-> [frozen] lm_head -> P(token)
    text  -> [frozen] Qwen -> h_llm(l) -----'                   `-> blank head    -> P(blank)

Measured on this setup (see SLIMDUCER.md):

  * AuT runs at 13 Hz and the 10 h transcripts tokenise to 3.05 Qwen tokens
    per second once lower-cased, so T/L = 4.3 and 76.5 % of frames carry no
    token. Blank is the majority class, which is why it gets its own head
    rather than a slot in the frozen vocabulary -- a reserved token's
    lm_head row points in an arbitrary direction that cannot be changed.
  * Consecutive duplicate tokens are 0.07 % of the corpus, so blank is not
    here to disambiguate repeats; it is here because frames outnumber
    tokens four to one.

Output factorisation (Hybrid Autoregressive Transducer, Variani et al.
2020):

    beta_t = sigmoid(w . h_t + b)          P(blank), d + 1 trained weights
    p_t    = softmax(lm_head(h_t))         token distribution, frozen

    blank frame  : log beta_t
    token y      : log(1 - beta_t) + log p_t[y]

Folding blank into the vocabulary softmax instead would put it in the same
normaliser as the words and the token distribution would no longer be the
frozen LLM's. Factorised, it is -- exactly.

Training alternates hard alignment and teacher forcing, in the style of
Viterbi training: align the reference against the current model, fit the
frame-level targets, realign. Because the alignment is fixed rather than
marginalised, nothing of shape (T, U, V) is ever materialised -- the loss
sees (T, V), which is what keeps this affordable against a 152 k
vocabulary.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

LLM_NAME = "Qwen/Qwen3-0.6B"
AUT_NAME = "Qwen/Qwen3-ASR-0.6B-hf"

# Frames whose alignment score is -inf. Not float("-inf"): the DP adds these
# and an inf - inf would make the backtrace NaN rather than merely unlikely.
NEG = -1e30


@dataclasses.dataclass
class SlimDucerOutput:
    loss: torch.Tensor
    blank_loss: torch.Tensor
    token_loss: torch.Tensor
    n_frames: int
    n_tokens: int


class JointMlp(nn.Module):
    """activation(W_llm h_llm + W_aut h_aut + b) -> LLM hidden size.

    The final width is not a free choice: the frozen lm_head consumes it,
    so the joint has to hand back exactly what the LLM would have handed
    its own output projection. `hidden` only sets the width of the
    intermediate space where the two modalities are added.

    `branch_norm` puts a LayerNorm on each input first, and it matters more
    than it looks. The two towers were never built to be added together:
    measured on a trained checkpoint, h_llm arrives with an L2 norm of 99.0
    against h_aut's 17.2, and even after training pushed W_aut (74.2) above
    W_llm (52.3) the LLM term still enters the sum 3.15x larger. GELU is
    close to linear at that magnitude, so the larger term sets the output
    and the smaller one reads as a perturbation -- which is exactly what
    the ablation showed, the LLM worth 16.2 points of frame accuracy
    against the audio's 8.35. Normalising first puts both paths on the same
    footing and lets training, not the two pre-trainings' unrelated scale
    conventions, decide the weighting.
    """

    def __init__(self, d_llm: int, d_aut: int, hidden: int,
                 activation: str = "gelu", branch_norm: bool = False):
        super().__init__()
        self.norm_llm = nn.LayerNorm(d_llm) if branch_norm else nn.Identity()
        self.norm_aut = nn.LayerNorm(d_aut) if branch_norm else nn.Identity()
        self.from_llm = nn.Linear(d_llm, hidden, bias=False)
        self.from_aut = nn.Linear(d_aut, hidden, bias=True)
        self.act = {"gelu": nn.GELU(), "relu": nn.ReLU(),
                    "silu": nn.SiLU(), "tanh": nn.Tanh()}[activation]
        self.out = nn.Linear(hidden, d_llm)

    def forward(self, h_llm: torch.Tensor,
                h_aut: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.from_llm(self.norm_llm(h_llm))
                                 + self.from_aut(self.norm_aut(h_aut))))


class SlimDucer(nn.Module):
    def __init__(self, llm_name: str = LLM_NAME, aut_name: str = AUT_NAME,
                 joint_hidden: int = 1024, activation: str = "gelu",
                 freeze_audio: bool = False,
                 branch_norm: bool = False,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import (AutoModelForCausalLM,
                                  Qwen3ASRForConditionalGeneration)

        # The text-only Qwen, NOT the language tower inside Qwen3-ASR: that
        # one has been fine-tuned for transcription, and the whole point
        # here is that an untouched text LLM is enough.
        llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=dtype)
        self.llm = llm.model            # embeddings + layers + final norm
        self.lm_head = llm.lm_head      # tied to the embeddings; frozen
        self.d_llm = llm.config.hidden_size
        self.vocab_size = llm.config.vocab_size

        asr = Qwen3ASRForConditionalGeneration.from_pretrained(aut_name,
                                                               dtype=dtype)
        # multi_modal_projector is deliberately dropped. That layer exists to
        # lift audio into the LLM's INPUT space, which is the coupling this
        # model replaces; what we want is the representation below it.
        self.audio_tower = asr.model.audio_tower
        self.d_aut = asr.config.audio_config.d_model
        self.n_mels = asr.config.audio_config.num_mel_bins
        self.max_mel = 3000            # feature extractor pads to 30 s
        del asr

        for p in self.llm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        if freeze_audio:
            for p in self.audio_tower.parameters():
                p.requires_grad = False

        self.joint = JointMlp(self.d_llm, self.d_aut, joint_hidden,
                              activation, branch_norm).to(dtype)
        self.blank_head = nn.Linear(self.d_llm, 1).to(dtype)

    # ---------------------------------------------------------------- utils

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # The tower returns PACKED output -- every utterance's valid frames
    # concatenated with nothing marking the seams -- so the split needs the
    # per-utterance frame count. Reverse-engineering it from the mel length
    # does not work: the staircase has period 8 over most of the range but
    # skips (measured 440->57 and 441->58, yet 540->70 and 541->71), so a
    # closed form that fits one stretch is wrong elsewhere, and a one-frame
    # error silently shifts an utterance's acoustics into its neighbour's
    # alignment. Probing the tower once per distinct mel length and caching
    # is exact, and the cost vanishes after the first epoch.
    _LEN_CACHE: dict[int, int] = {}

    def audio_out_len(self, mel_len: int) -> int:
        mel_len = int(mel_len)
        hit = self._LEN_CACHE.get(mel_len)
        if hit is not None:
            return hit
        p = next(self.audio_tower.parameters())
        x = torch.zeros((1, self.n_mels, self.max_mel), dtype=p.dtype,
                        device=p.device)
        m = torch.zeros((1, self.max_mel), dtype=torch.bool, device=p.device)
        m[0, :mel_len] = True
        with torch.no_grad():
            out = self.audio_tower(x, m)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        n = int(h.shape[-2] if h.dim() == 3 else h.shape[0])
        self._LEN_CACHE[mel_len] = n
        return n

    def encode_audio(self, input_features: torch.Tensor,
                     input_features_mask: torch.Tensor) -> list[torch.Tensor]:
        """(B, n_mels, S) -> list of (T_b, d_aut), one per utterance.

        The mask is not optional bookkeeping. The feature extractor pads
        every input to 30 s, so an all-ones mask makes every utterance 390
        frames of which most are silence, and the aligner will put tokens
        there.
        """
        out = self.audio_tower(input_features, input_features_mask)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        lens = [self.audio_out_len(int(m.sum())) for m in input_features_mask]
        if h.dim() == 3:                      # some versions keep the batch
            return [h[i, :n] for i, n in enumerate(lens)]
        assert sum(lens) == h.shape[0], (lens, h.shape)
        return list(torch.split(h, lens, dim=0))

    def encode_text(self, labels: torch.Tensor,
                    bos_id: int) -> torch.Tensor:
        """(B, L) -> (B, L+1, d_llm), the frozen LLM's states.

        Position l holds the state after l tokens, so a frame that has
        emitted l tokens so far reads row l -- including row 0, the state
        after BOS alone, which every frame before the first emission uses.
        """
        b = labels.shape[0]
        bos = labels.new_full((b, 1), bos_id)
        seq = torch.cat([bos, labels.clamp(min=0)], dim=1)      # (B, L+1)
        with torch.no_grad():
            out = self.llm(input_ids=seq)
        return out.last_hidden_state

    # ------------------------------------------------------------ alignment

    @torch.no_grad()
    def align(self, h_aut: torch.Tensor, h_llm: torch.Tensor,
              t_len: int, l_len: int, labels: torch.Tensor,
              chunk: int = 8) -> torch.Tensor:
        """Monotonic Viterbi alignment for ONE utterance.

        Returns (T,) long: -1 for a blank frame, otherwise the index of the
        token emitted at that frame. Each of the L tokens is emitted exactly
        once and in order, so the returned path is a decomposition of the
        reference, not a relabelling of it.

        The grid is (T, L+1) because the emission score at frame t for token
        l depends on the LLM state after l-1 tokens -- the model is
        autoregressive, so "which token" and "how many so far" are not
        separable. That is also why this costs T*L joint evaluations rather
        than T: the same frame is scored against every reachable prefix
        length.
        """
        device = h_aut.device
        # The caller passes the batch-padded state, whose first axis is the
        # LONGEST transcript in the batch; this utterance owns only the
        # first l_len + 1 rows. Slicing here rather than at the call site
        # keeps every caller from having to remember it.
        h_llm = h_llm[:l_len + 1]
        # (T, L+1, d) is the whole cost. Chunking over t keeps the 152 k-wide
        # logits from being materialised for the entire utterance at once.
        log_blank = torch.empty((t_len, l_len + 1), device=device,
                                dtype=torch.float32)
        log_emit = torch.full((t_len, l_len + 1), NEG, device=device,
                              dtype=torch.float32)
        for s in range(0, t_len, chunk):
            e = min(s + chunk, t_len)
            a = h_aut[s:e].unsqueeze(1).expand(-1, l_len + 1, -1)   # (c,L+1,da)
            m = h_llm.unsqueeze(0).expand(e - s, -1, -1)            # (c,L+1,dl)
            h = self.joint(m, a)
            beta = torch.sigmoid(self.blank_head(h).squeeze(-1)).float()
            log_blank[s:e] = torch.log(beta.clamp(min=1e-9))
            logits = self.lm_head(h).float()                        # (c,L+1,V)
            lsm = torch.log_softmax(logits, dim=-1)
            # Emitting token l at this frame is only defined for l < L, and
            # it is read from the row for "l tokens already emitted".
            idx = labels[:l_len].view(1, l_len, 1).expand(e - s, -1, -1)
            log_emit[s:e, :l_len] = (
                torch.log((1.0 - beta[:, :l_len]).clamp(min=1e-9))
                + lsm[:, :l_len].gather(-1, idx).squeeze(-1))

        score = torch.full((l_len + 1,), NEG, device=device,
                           dtype=torch.float32)
        score[0] = 0.0
        back = torch.zeros((t_len, l_len + 1), dtype=torch.bool, device=device)
        for t in range(t_len):
            stay = score + log_blank[t]
            move = torch.full_like(score, NEG)
            move[1:] = score[:-1] + log_emit[t, :l_len]
            back[t] = move > stay
            score = torch.where(back[t], move, stay)

        path = torch.full((t_len,), -1, dtype=torch.long, device=device)
        l = l_len
        for t in range(t_len - 1, -1, -1):
            if l > 0 and back[t, l]:
                path[t] = l - 1
                l -= 1
        # A reference that cannot fit (more tokens than frames) would leave
        # l > 0 here; the caller should drop such utterances rather than
        # train on a truncated transcript.
        if l != 0:
            path[:] = -1
        return path

    # -------------------------------------------------------------- forward

    def forward(self, input_features: torch.Tensor,
                input_features_mask: torch.Tensor,
                labels: torch.Tensor, label_lens: torch.Tensor,
                alignment: Sequence[torch.Tensor], bos_id: int
                ) -> SlimDucerOutput:
        """Teacher forcing against a fixed alignment.

        `alignment[b]` is what `align` returns: -1 for blank, else a token
        index. The LLM is fed the REFERENCE prefix, never the argmax, so a
        wrong frame cannot poison the conditioning of the frames after it.
        """
        h_aut_list = self.encode_audio(input_features, input_features_mask)
        h_llm_all = self.encode_text(labels, bos_id)

        blank_terms, token_terms = [], []
        n_frames = n_tokens = 0
        for b, path in enumerate(alignment):
            t_len = int(path.shape[0])
            if t_len == 0 or bool((path < 0).all()) and int(label_lens[b]) > 0:
                continue
            # n_before[t] = tokens emitted strictly before frame t, which is
            # exactly the row of the LLM state this frame conditions on.
            emitted = (path >= 0).long()
            n_before = torch.cat([emitted.new_zeros(1),
                                  emitted.cumsum(0)[:-1]])
            h_llm = h_llm_all[b].index_select(0, n_before)       # (T, d_llm)
            h = self.joint(h_llm, h_aut_list[b][:t_len])         # (T, d_llm)

            logit_blank = self.blank_head(h).squeeze(-1).float() # (T,)
            is_blank = (path < 0)
            blank_terms.append(F.binary_cross_entropy_with_logits(
                logit_blank, is_blank.to(logit_blank.dtype), reduction="sum"))

            emit_idx = (~is_blank).nonzero(as_tuple=True)[0]
            if emit_idx.numel():
                tgt = labels[b].index_select(0, path[emit_idx])
                logits = self.lm_head(h.index_select(0, emit_idx)).float()
                token_terms.append(F.cross_entropy(logits, tgt,
                                                   reduction="sum"))
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

    # --------------------------------------------------------------- decode

    @torch.no_grad()
    def decode(self, input_features: torch.Tensor,
               input_features_mask: torch.Tensor, bos_id: int,
               t_lens: Optional[torch.Tensor] = None,
               blank_threshold: float = 0.5,
               max_tokens: int = 512) -> list[list[int]]:
        """Frame-synchronous greedy decode, one utterance at a time.

        The LLM advances only when a token is emitted, so it runs L times
        and not T. That is the whole efficiency argument for this model, and
        `test_llm_runs_once_per_token` pins it.

        `blank_threshold` moves the deletion/insertion operating point
        without retraining -- the emission decision is a single sigmoid, so
        it is a knob rather than a property of the weights.
        """
        h_aut_list = self.encode_audio(input_features, input_features_mask)
        b_size = len(h_aut_list)
        hyps: list[list[int]] = []
        for b in range(b_size):
            t_len = (int(t_lens[b]) if t_lens is not None
                     else h_aut_list[b].shape[0])
            dev = h_aut_list[b].device
            ids = torch.tensor([[bos_id]], device=dev)
            out = self.llm(input_ids=ids, use_cache=True)
            h_llm = out.last_hidden_state[:, -1]                 # (1, d_llm)
            cache = out.past_key_values
            hyp: list[int] = []
            for t in range(t_len):
                h = self.joint(h_llm, h_aut_list[b][t:t + 1])
                if torch.sigmoid(self.blank_head(h)).item() >= blank_threshold:
                    continue                                     # blank: no LLM
                tok = int(self.lm_head(h).argmax(-1).item())
                hyp.append(tok)
                if len(hyp) >= max_tokens:
                    break
                nxt = torch.tensor([[tok]], device=dev)
                out = self.llm(input_ids=nxt, past_key_values=cache,
                               use_cache=True)
                h_llm = out.last_hidden_state[:, -1]
                cache = out.past_key_values
            hyps.append(hyp)
        return hyps
