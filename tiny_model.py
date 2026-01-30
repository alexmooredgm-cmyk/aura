# TinyMind: a very small autoregressive "mind" implemented with tinygrad.
# Architecture: embedding lookup -> simple RNN cell -> linear output logits.
# Uses tinygrad.Tensor for autograd and supports a basic policy-gradient update.
import numpy as np
from tinygrad.tensor import Tensor

def randn(shape, scale=0.1):
    return Tensor(np.random.randn(*shape).astype(np.float32) * scale)

class TinyMind:
    def __init__(self, vocab_size, emb_size=32, hidden_size=64, context_size=16, use_value_head=False):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.context_size = context_size

        # Parameters (all tinygrad Tensors)
        self.emb = randn((vocab_size, emb_size))
        self.W_in = randn((emb_size, hidden_size))
        self.W_h = randn((hidden_size, hidden_size))
        self.W_ec = randn((hidden_size, hidden_size))  # projector for encoder_ctx
        self.W_ctx = randn((context_size, hidden_size))
        self.b_rnn = randn((hidden_size,))  # bias vector

        self.W_out = randn((hidden_size, vocab_size))
        self.b_out = randn((vocab_size,))

        # ensure requires_grad True
        for p in self.parameters():
            p.requires_grad = True

    def parameters(self):
        return [self.emb, self.W_in, self.W_h, self.W_ec, self.W_ctx, self.b_rnn, self.W_out, self.b_out]

    def encode_input(self, input_ids):
        # input_ids: python list of ints (with BOS/EOS). Return encoder_ctx Tensor shape (hidden,)
        if len(input_ids) == 0:
            return Tensor(np.zeros((self.hidden_size,), dtype=np.float32))
        # lookup embeddings and average
        embs = [self.emb[int(i)] for i in input_ids]
        stacked = embs[0]
        if len(embs) > 1:
            for e in embs[1:]:
                stacked = stacked + e
        avg = stacked * (1.0 / float(len(embs)))  # Tensor
        # project average emb -> hidden-size context
        encoder_ctx = avg.matmul(self.W_in)  # reuse W_in as a simple projector
        return encoder_ctx  # Tensor (hidden,)

    def rnn_step(self, prev_token_id, prev_hidden, encoder_ctx, context_vec):
        # prev_token_id: int, prev_hidden: Tensor (hidden,) or None
        prev_emb = self.emb[int(prev_token_id)]  # (emb,)
        h_in = prev_emb.matmul(self.W_in)
        h_prev = prev_hidden if prev_hidden is not None else Tensor(np.zeros((self.hidden_size,), dtype=np.float32))
        h_rec = h_prev.matmul(self.W_h)
        h_ec = encoder_ctx.matmul(self.W_ec)
        h_ctx = context_vec.matmul(self.W_ctx)
        h = (h_in + h_rec + h_ec + h_ctx + self.b_rnn).tanh()
        logits = h.matmul(self.W_out) + self.b_out  # (vocab,)
        return logits, h

    def greedy_decode(self, encoder_ctx, context_vec, max_len=20, sos_id=2, eos_id=3, sample=True):
        # encoder_ctx: Tensor (hidden,), context_vec: Tensor (context_size,)
        prev = sos_id
        hidden = None
        out_ids = []
        log_probs = []
        for _ in range(max_len):
            logits, hidden = self.rnn_step(prev, hidden, encoder_ctx, context_vec)
            # convert logits to numpy probs for sampling/greedy
            logits_np = logits.numpy()
            # numerically stable softmax
            exps = np.exp(logits_np - np.max(logits_np))
            probs = exps / (exps.sum() + 1e-12)
            if sample:
                nxt = int(np.random.choice(len(probs), p=probs))
            else:
                nxt = int(np.argmax(probs))
            out_ids.append(nxt)
            # compute logprob as scalar
            lp = float(np.log(probs[nxt] + 1e-12))
            log_probs.append(lp)
            prev = nxt
            if nxt == eos_id:
                break
        return out_ids, log_probs  # lists

    # Convenience: get numpy copy of a tinygrad Tensor
    # (not needed in many places; call .numpy() on Tensor)