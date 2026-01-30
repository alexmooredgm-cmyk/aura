# Main server: Flask + SocketIO. Hosts the UI, handles player messages, runs model generation,
# computes reward, and performs an online optimizer step.
import os
import time
import threading
import json
import torch
import torch.nn.functional as F
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from tokenizer import SimpleTokenizer, BOS, EOS
from model import MindNet
from memory import SessionMemory
from rewards import Rewarder
import numpy as np

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Initialize components
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Prepare tokenizer with small default vocab
if not os.path.exists("vocab.json"):
    # initialize tiny starter vocab (extendable)
    seed_texts = ["hello", "i am a mind", "how are you?", "i like learning", "what is your name?"]
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab_from_texts(seed_texts, max_size=2000)
    tokenizer.save("vocab.json")
else:
    tokenizer = SimpleTokenizer.load("vocab.json")

vocab_size = tokenizer.size()
# Model hyperparams
EMB = 128
HID = 256
CONTEXT_SIZE = 32

model = MindNet(vocab_size, emb_size=EMB, hidden_size=HID, context_size=CONTEXT_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Very small function to create a context vector (time, emotion placeholders)
def make_context_vector(timestamp=None, emotion=0.0):
    # returns torch tensor (1, context_size)
    vec = np.zeros(CONTEXT_SIZE, dtype=np.float32)
    t = time.time() if timestamp is None else timestamp
    # simple normalized time-of-day feature
    tod = (t % 86400) / 86400.0
    vec[0] = tod
    vec[1] = float(emotion)
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# In-memory sessions (map sid -> SessionMemory)
SESSIONS = {}
REWARDER = Rewarder()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("connect")
def on_connect():
    sid = str(request.sid) if False else None
    emit("connected", {"msg":"connected"})

@socketio.on("start_session")
def start_session(data):
    # data can contain optional session_id; we create new session
    sid = data.get("session_id", str(time.time()))
    mem = SessionMemory(device=DEVICE, memory_dim=128)
    SESSIONS[sid] = mem
    emit("session_started", {"session_id": sid})

@socketio.on("player_message")
def handle_player_message(data):
    """
    data: {
      session_id: str,
      text: str,
      feedback: 0|1|-1 (optional: explicit feedback for previous AI turn),
      correction: optional corrected response text when user corrects
    }
    """
    sid = data.get("session_id")
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionMemory(device=DEVICE, memory_dim=128)
    mem = SESSIONS[sid]
    text = data.get("text", "")
    feedback = data.get("feedback", 0)
    correction = data.get("correction", None)

    # mark engagement timing
    REWARDER.mark_player_message()

    # Encode input
    input_ids = tokenizer.encode(text, add_bos_eos=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        encoder_ctx = model.encode_input(input_tensor)

    # construct context vec
    context_vec = make_context_vector()

    # set hidden state from memory if present
    if mem.get_hidden() is not None:
        try:
            hidden_state = mem.get_hidden()  # (1,1,hidden)
        except Exception:
            hidden_state = None
    else:
        hidden_state = None

    # Generate a response (sampled to encourage novelty)
    model.eval()
    with torch.no_grad():
        outputs, log_probs = model.greedy_decode(encoder_ctx, context_vec, max_len=60, sample=True)
    ai_tokens = [t for t in outputs[0]]
    ai_text = tokenizer.decode(ai_tokens)
    # compute simple internal metrics
    # confidence: mean max-softmax over generated tokens (approx)
    # For demo, recompute token probs quickly by re-running decode deterministically to get probs
    model.eval()
    # We'll approximate confidence as normalized avg logprob
    avg_logprob = float(log_probs.mean().cpu().numpy()) if log_probs.numel() else -5.0
    confidence = float((avg_logprob + 5.0) / 5.0)
    confidence = max(0.0, min(1.0, confidence))

    # novelty: use rewarder
    novelty = REWARDER.novelty_reward(ai_tokens, mem)
    engagement = REWARDER.engagement_reward()
    # placeholder embeddings for consistency: use encoder_ctx as embedding
    embedding = encoder_ctx.squeeze(0).detach()
    recent_embeddings = []  # not computing detailed embeddings here; placeholder
    consistency = 0.5

    # compute combined reward (for updating the model)
    combined_r = REWARDER.combined_reward(feedback, engagement, novelty, consistency)

    # prepare training update: REINFORCE-style
    # we will perform a single-step update: maximize reward * sum(log probs) (i.e., minimize -r * sum log probs)
    model.train()
    # Recompute generation with gradient tracking so we can compute log-probs
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    encoder_ctx = model.encode_input(input_tensor)
    context_vec = make_context_vector()
    # run a generation loop that replays the tokens we sampled (teacher-forcing over the sampled sequence
    # to get logprobs). For simplicity, feed the generated tokens back.
    prev = torch.tensor([tokenizer.token_to_id[BOS]], dtype=torch.long, device=DEVICE)
    hidden = mem.get_hidden() if mem.get_hidden() is not None else torch.zeros(1,1,encoder_ctx.size(1), device=DEVICE)
    logprob_sum = 0.0
    generated_ids = []
    for t_id in ai_tokens:
        t_id_t = torch.tensor([t_id], dtype=torch.long, device=DEVICE)
        logits, hidden, value = model.forward_decode_step(prev, hidden, encoder_ctx, context_vec)
        logp = F.log_softmax(logits, dim=-1)
        sel_logp = logp.gather(1, t_id_t.unsqueeze(-1)).squeeze(-1)
        logprob_sum = logprob_sum + sel_logp
        prev = t_id_t
        generated_ids.append(t_id)
        # stop if EOS
        if t_id == tokenizer.token_to_id[EOS]:
            break
    # policy loss (negative reward times logprob)
    policy_loss = - (combined_r * logprob_sum).mean()
    # if user provided a correction, add supervised teacher loss (cross-entropy) for target tokens
    sup_loss = torch.tensor(0.0, device=DEVICE)
    if correction:
        target_ids = tokenizer.encode(correction, add_bos_eos=True)
        # simple teacher forcing over correction tokens
        prev = torch.tensor([tokenizer.token_to_id[BOS]], dtype=torch.long, device=DEVICE)
        hidden2 = mem.get_hidden() if mem.get_hidden() is not None else torch.zeros(1,1,encoder_ctx.size(1), device=DEVICE)
        ce_losses = []
        for tgt in target_ids[1:]:  # ignore bos
            tgt_t = torch.tensor([tgt], dtype=torch.long, device=DEVICE)
            logits, hidden2, _ = model.forward_decode_step(prev, hidden2, encoder_ctx, context_vec)
            ce = F.cross_entropy(logits, tgt_t, reduction="none")
            ce_losses.append(ce)
            prev = tgt_t
        if ce_losses:
            sup_loss = torch.stack(ce_losses).mean()
    total_loss = policy_loss + 0.5 * sup_loss

    optimizer.zero_grad()
    total_loss.backward()
    # small gradient clipping to avoid instability
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # after update, store new hidden state into memory
    mem.set_hidden(hidden.detach().clone())
    mem.push_turn({"player": text, "ai": ai_text, "timestamp": time.time(),
                   "player_tokens": input_ids, "ai_tokens": generated_ids})
    # create mental-state metrics
    mental_state = {
        "curiosity": float(1.0 - novelty),  # naive: novelty -> curiosity inverse (tweakable)
        "confidence": confidence,
        "confusion": float(1.0 - confidence)
    }

    # send response
    emit("ai_response", {"text": ai_text, "mental_state": mental_state, "session_id": sid})

if __name__ == "__main__":
    # ensure template/static directories exist
    socketio.run(app, host="0.0.0.0", port=5000)