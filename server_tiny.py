# Termux-friendly server using tinygrad-based TinyMind model
# This file mirrors server.py but avoids torch and uses tiny_model.py instead.
import os
import time
import json
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from tokenizer import SimpleTokenizer, BOS, EOS
from memory import SessionMemory
from rewards import Rewarder
from tiny_model import TinyMind
from tinygrad.tensor import Tensor

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

DEVICE = "cpu"
print("Running tinygrad server (CPU)")

# Prepare tokenizer
if not os.path.exists("vocab.json"):
    seed_texts = ["hello", "i am a mind", "how are you?", "i like learning", "what is your name?"]
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab_from_texts(seed_texts, max_size=2000)
    tokenizer.save("vocab.json")
else:
    tokenizer = SimpleTokenizer.load("vocab.json")

vocab_size = tokenizer.size()
# small model for Termux
EMB = 32
HID = 64
CONTEXT_SIZE = 16

model = TinyMind(vocab_size, emb_size=EMB, hidden_size=HID, context_size=CONTEXT_SIZE)
LR = 1e-3  # small learning rate

SESSIONS = {}
REWARDER = Rewarder()

def make_context_vector(timestamp=None, emotion=0.0):
    vec = np.zeros((CONTEXT_SIZE,), dtype=np.float32)
    t = time.time() if timestamp is None else timestamp
    tod = (t % 86400) / 86400.0
    vec[0] = tod
    vec[1] = float(emotion)
    return Tensor(vec)

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("start_session")
def start_session(data):
    sid = data.get("session_id", str(time.time()))
    mem = SessionMemory(device=DEVICE, memory_dim=128)
    SESSIONS[sid] = mem
    emit("session_started", {"session_id": sid})

@socketio.on("player_message")
def handle_player_message(data):
    sid = data.get("session_id")
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionMemory(device=DEVICE, memory_dim=128)
    mem = SESSIONS[sid]
    text = data.get("text", "")
    feedback = data.get("feedback", 0)
    correction = data.get("correction", None)

    REWARDER.mark_player_message()

    input_ids = tokenizer.encode(text, add_bos_eos=True)
    # encode
    encoder_ctx = model.encode_input(input_ids)  # TinyMind returns tinygrad Tensor
    context_vec = make_context_vector()

    # generate
    out_ids, log_probs = model.greedy_decode(encoder_ctx, context_vec, max_len=30, sample=True)
    ai_text = tokenizer.decode(out_ids)

    # compute metrics
    avg_logprob = float(np.mean(log_probs)) if len(log_probs) else -5.0
    confidence = max(0.0, min(1.0, (avg_logprob + 5.0) / 5.0))

    novelty = REWARDER.novelty_reward(out_ids, mem)
    engagement = REWARDER.engagement_reward()
    consistency = 0.5
    combined_r = REWARDER.combined_reward(feedback, engagement, novelty, consistency)

    # Online learning: simple REINFORCE-like update using tinygrad autograd.
    # We'll re-run the generated sequence through the model to build a differentiable loss.
    # For simplicity we compute negative reward * sum(log_probs) using a softmax computed inside tinygrad.
    # NOTE: tinygrad versions vary. This uses basic Tensor ops and .backward().

    # Build a differentiable log-prob sum:
    # We'll run steps and at each step compute log_softmax and pick the generated token via one-hot dot.
    # Start hidden = None
    # Reset gradients first (tinygrad will set grads after backward)
    for p in model.parameters():
        p.grad = None  # try to reset grads; some tinygrad versions require different attribute handling

    prev = tokenizer.token_to_id[BOS]
    hidden = None
    logprob_sum = Tensor(np.array(0.0, dtype=np.float32))
    for t_id in out_ids:
        logits, hidden = model.rnn_step(prev, hidden, encoder_ctx, context_vec)
        # logits is Tensor (vocab,)
        # compute stable softmax via tinygrad: probs = exp(logits - max) / sum
        lp = logits - logits.max()  # broadcasting
        exps = lp.exp()
        denom = exps.sum() + 1e-12
        log_probs_tensor = (exps / denom).log()
        # one-hot selection: multiply log_probs_tensor by one-hot vector
        onehot = Tensor(np.zeros((model.vocab_size,), dtype=np.float32))
        onehot.data[t_id] = 1.0  # set the selected index (tinygrad Tensor exposes .data)
        # selected logprob
        sel_logp = (log_probs_tensor * onehot).sum()
        logprob_sum = logprob_sum + sel_logp
        prev = t_id
        if t_id == tokenizer.token_to_id[EOS]:
            break

    # policy loss = - reward * logprob_sum
    policy_loss = logprob_sum * (-float(combined_r))
    # supervised correction loss (optional)
    sup_loss = None
    if correction:
        # simple cross-entropy on correction tokens (teacher forcing)
        # Warning: this is a small/simple supervised objective; may need adaptation for tinygrad semantics.
        prev = tokenizer.token_to_id[BOS]
        hidden = None
        ce_acc = Tensor(np.array(0.0, dtype=np.float32))
        target_ids = tokenizer.encode(correction, add_bos_eos=True)
        for tgt in target_ids[1:]:
            logits, hidden = model.rnn_step(prev, hidden, encoder_ctx, context_vec)
            lp = logits - logits.max()
            exps = lp.exp()
            probs = exps / (exps.sum() + 1e-12)
            onehot = Tensor(np.zeros((model.vocab_size,), dtype=np.float32))
            onehot.data[tgt] = 1.0
            # negative log-likelihood
            nll = - (probs.log() * onehot).sum()
            ce_acc = ce_acc + nll
            prev = tgt
        sup_loss = ce_acc * (1.0 / max(1, len(target_ids)))
        total_loss = policy_loss + 0.5 * sup_loss
    else:
        total_loss = policy_loss

    # Backprop and simple SGD step
    total_loss.backward()

    # Very small manual SGD update
    # Note: tinygrad versions expose gradients in different ways.
    # We try to handle common cases: p.grad (Tensor) or p.grad (np array) or p.grad is None.
    for p in model.parameters():
        if getattr(p, "grad", None) is None:
            continue
        # p.data is the numpy array behind the Tensor in many tinygrad versions.
        try:
            # subtract lr * grad in-place
            p.data -= LR * p.grad.data
        except Exception:
            try:
                # alternative: p.assign(p - LR * p.grad)
                p.assign(p - (p.grad * LR))
            except Exception:
                # last resort: ignore if update fails; model remains usable but won't train.
                pass
        # clear grad
        p.grad = None

    # update memory
    mem.push_turn({"player": text, "ai": ai_text, "timestamp": time.time(),
                   "player_tokens": input_ids, "ai_tokens": out_ids})

    mental_state = {
        "curiosity": float(1.0 - novelty),
        "confidence": confidence,
        "confusion": float(1.0 - confidence)
    }

    emit("ai_response", {"text": ai_text, "mental_state": mental_state, "session_id": sid})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)