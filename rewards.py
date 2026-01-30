# Reward computation helpers: combine feedback, engagement, novelty, consistency into a scalar reward (0..1).
import math
import time
import numpy as np
from collections import deque
from scipy.spatial.distance import cosine

class Rewarder:
    def __init__(self):
        self.last_player_time = None
        self.reply_times = deque(maxlen=200)

    def mark_player_message(self):
        now = time.time()
        if self.last_player_time is not None:
            delta = now - self.last_player_time
            self.reply_times.append(delta)
        self.last_player_time = now

    def engagement_reward(self):
        # shorter reply time -> higher engagement (bounded)
        if not self.reply_times:
            return 0.5
        avg = np.mean(self.reply_times)
        # map [0.5s, 60s] -> [1.0, 0.0]
        r = max(0.0, min(1.0, 1.0 - (avg - 0.5) / (60.0 - 0.5)))
        return float(r)

    def novelty_reward(self, token_ids, session_memory):
        # compute novelty as inverse average token frequency
        if not token_ids:
            return 0.0
        freqs = [session_memory.token_freq.get(t, 0) for t in token_ids]
        avg_freq = (sum(freqs) / len(freqs)) + 1.0
        # map inverse frequency into [0,1]
        return float(1.0 / math.log(avg_freq + 1.5))

    def consistency_reward(self, embedding, recent_embeddings):
        # similarity with recent embeddings
        if len(recent_embeddings) == 0:
            return 0.5
        # compute cosine similarity from -1..1 -> map to 0..1
        cos = 0.0
        # use torch tensors (embedding as numpy)
        try:
            import torch
            emb = embedding / (embedding.norm() + 1e-9)
            others = torch.stack([o / (o.norm()+1e-9) for o in recent_embeddings])
            cos = float((others @ emb).mean().item())
        except Exception:
            cos = 0.0
        return float((cos + 1.0) / 2.0)

    def combined_reward(self, player_feedback, engagement, novelty, consistency, weights=None):
        # player_feedback: -1 (dislike), 0 (none), +1 (like)
        if weights is None:
            weights = {"feedback": 0.5, "engagement": 0.2, "novelty": 0.15, "consistency": 0.15}
        feedback_score = 0.5 if player_feedback == 0 else (1.0 if player_feedback > 0 else 0.0)
        r = (weights["feedback"] * feedback_score +
             weights["engagement"] * engagement +
             weights["novelty"] * novelty +
             weights["consistency"] * consistency)
        # bound
        return max(0.0, min(1.0, r))