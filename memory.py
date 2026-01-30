# Session memory management: stores hidden state, episodic buffer, token frequencies, and learned memory vector.
import numpy as np
import torch

class SessionMemory:
    def __init__(self, device="cpu", memory_dim=128):
        self.device = device
        self.hidden_state = None  # torch tensor (1, batch=1, hidden)
        self.episodic = []  # list of conversation turns (dicts)
        self.token_freq = {}
        self.learned_memory = torch.zeros(memory_dim, device=device)

    def set_hidden(self, hidden):
        # expects torch tensor
        self.hidden_state = hidden.detach().clone()

    def get_hidden(self):
        return None if self.hidden_state is None else self.hidden_state.detach().clone()

    def push_turn(self, turn):
        # turn: dict { 'player':str, 'ai':str, 'timestamp':float, ...}
        self.episodic.append(turn)
        # maintain limited size
        if len(self.episodic) > 200:
            self.episodic.pop(0)
        # update token frequencies if available
        for t in (turn.get("ai_tokens", []) + turn.get("player_tokens", [])):
            self.token_freq[t] = self.token_freq.get(t, 0) + 1

    def get_recent_texts(self, n=10):
        return [t for turn in self.episodic[-n:] for t in (turn.get("ai", ""), turn.get("player", ""))]

    def similarity_score(self, embedding, other_embeddings):
        # naive cosine against last few embeddings
        if len(other_embeddings) == 0:
            return 0.0
        emb = embedding / (embedding.norm() + 1e-9)
        others = torch.stack([o / (o.norm()+1e-9) for o in other_embeddings], dim=0)
        sims = torch.matmul(others, emb)
        return float(sims.mean().item())