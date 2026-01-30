# Core PyTorch model: embedding + encoder GRU + decoder GRU for autoregressive generation
# Includes a policy head (token logits) and an optional value head for expected reward.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MindNet(nn.Module):
    def __init__(self, vocab_size, emb_size=128, hidden_size=256, context_size=32, use_value_head=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        # encoder to read player input (simple bidirectional)
        self.encoder = nn.GRU(emb_size, hidden_size//2, batch_first=True, bidirectional=True)
        # small context projector (for time, emotion tokens)
        self.context_proj = nn.Linear(context_size, hidden_size)
        # decoder GRU that produces tokens autoregressively
        self.decoder = nn.GRU(emb_size + hidden_size + context_size, hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.use_value_head = use_value_head
        if use_value_head:
            self.value_head = nn.Linear(hidden_size, 1)

    def encode_input(self, input_ids, input_lengths=None):
        # input_ids: (batch, seq)
        emb = self.emb(input_ids)  # (batch, seq, emb)
        packed_out, h_n = self.encoder(emb)  # h_n: (num_layers*2, batch, hidden//2)
        # combine bidir states
        # reshape to (batch, hidden)
        if h_n.dim() == 3:
            h_n = h_n.view(h_n.size(1), -1)
        return h_n  # (batch, hidden)

    def forward_decode_step(self, prev_token_id, hidden_state, encoder_context, context_vec):
        """
        Single decoding timestep.
        prev_token_id: (batch,)
        hidden_state: (1, batch, hidden)
        encoder_context: (batch, hidden)  # repeated as context
        context_vec: (batch, context_size)
        Returns logits (batch, vocab), new_hidden
        """
        emb_t = self.emb(prev_token_id).unsqueeze(1)  # (batch,1,emb)
        # concat encoder_context and context_vec to token input
        ctx = torch.cat([encoder_context, context_vec], dim=-1).unsqueeze(1)  # (batch,1,hidden+context)
        decoder_input = torch.cat([emb_t, ctx], dim=-1)  # (batch,1, emb + hidden + context)
        out, new_h = self.decoder(decoder_input, hidden_state)  # out: (batch,1,hidden)
        out = out.squeeze(1)  # (batch, hidden)
        logits = self.output_proj(out)  # (batch, vocab)
        value = self.value_head(out).squeeze(-1) if self.use_value_head else None
        return logits, new_h, value

    def greedy_decode(self, encoder_ctx, context_vec, max_len=40, sos_id=2, eos_id=3, temperature=1.0, sample=False):
        # encoder_ctx: (batch, hidden), context_vec: (batch, context_size)
        batch = encoder_ctx.size(0)
        device = encoder_ctx.device
        hidden = torch.zeros(1, batch, encoder_ctx.size(1), device=device)
        prev = torch.full((batch,), sos_id, dtype=torch.long, device=device)
        outputs = []
        log_probs = []
        for t in range(max_len):
            logits, hidden, _ = self.forward_decode_step(prev, hidden, encoder_ctx, context_vec)
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(probs, dim=-1)
            outputs.append(next_token.cpu().numpy())
            # store logprob for RL updates
            selected_logprob = torch.log(probs.gather(1, next_token.unsqueeze(-1)).squeeze(-1) + 1e-12)
            log_probs.append(selected_logprob)
            prev = next_token
            if ((next_token == eos_id).all()):
                break
        # stack
        if len(outputs) == 0:
            return [[]], torch.zeros(1)
        outputs = list(map(list, zip(*outputs)))  # batch list of lists
        log_probs = torch.stack(log_probs, dim=1)  # (batch, seq)
        return outputs, log_probs
