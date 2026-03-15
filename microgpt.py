"""
microgpt_digits.py

Pure-Python microGPT adapted for an 8x8 digits-style dataset stored in ascii_dataset.txt.

Expected dataset format (IMPORTANT):
- Each example/document starts with a conditioning line: @<digit>
- Followed by 8 lines (each line length 8), lines separated by '\n'
- Documents are separated by a blank line => '\n\n'

Example (one document):
@5
0123ABCD
0123ABCD
...
(8 lines total)

(blank line)
(next document)

Outputs:
- Trains a tiny GPT-like model
- Saves ascii_model.pkl with weights + vocab + hyperparams
- Prints some generated 8x8 samples
"""

import os
import math
import random
import pickle

random.seed(42)

# -----------------------------------------------------------------------------
# 1) Load dataset: docs separated by blank line
# -----------------------------------------------------------------------------
DATA_PATH = "ascii_dataset.txt"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"{DATA_PATH} not found.\n"
        f"Create it first (digits dataset script) so each doc is 8 lines, and docs are separated by blank lines."
    )

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = f.read()

# Split on blank lines
docs = [doc.strip() for doc in data.strip().split("\n\n") if doc.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Quick sanity checks (won't stop training, but helps debug)
if len(docs) > 0:
    print("example doc (first 120 chars, escaped newlines):",
          docs[0][:120].replace("\n", "\\n"))

# -----------------------------------------------------------------------------
# 2) Tokenizer: character-level
# -----------------------------------------------------------------------------
uchars = sorted(set("".join(docs)))
BOS = len(uchars)               # special Beginning-of-Sequence token id
vocab_size = len(uchars) + 1
stoi = {ch: i for i, ch in enumerate(uchars)}  # faster than uchars.index
print(f"vocab size: {vocab_size}")
print("newline in vocab?", "\n" in uchars)


# Optional sanity checks for conditioned dataset format.
# This keeps backward compatibility with old datasets, while warning if
# a mixed/unexpected format is detected.
def _is_conditioned_doc(doc):
    lines = doc.splitlines()
    if not lines:
        return False
    if len(lines[0]) != 2 or lines[0][0] != "@" or not lines[0][1].isdigit():
        return False
    return True

conditioned_docs = sum(1 for d in docs if _is_conditioned_doc(d))
if conditioned_docs == len(docs):
    print("dataset format: conditioned (@<digit> header present)")
elif conditioned_docs == 0:
    print("dataset format: unconditioned (no @<digit> headers)")
else:
    print(
        f"warning: mixed dataset format ({conditioned_docs}/{len(docs)} conditioned docs)")

# -----------------------------------------------------------------------------
# 3) Autograd: scalar reverse-mode automatic differentiation
# -----------------------------------------------------------------------------
class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # other is a Python float/int exponent
        return Value(self.data ** other, (self,), (other * (self.data ** (other - 1.0)),))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def relu(self):
        out = self.data if self.data > 0.0 else 0.0
        grad = 1.0 if self.data > 0.0 else 0.0
        return Value(out, (self,), (grad,))

    def __neg__(self): return self * -1.0
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1.0)
    def __rtruediv__(self, other): return other * (self ** -1.0)

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# -----------------------------------------------------------------------------
# 4) Tiny GPT hyperparameters (tuned for 8x8 docs)
# -----------------------------------------------------------------------------
# An 8x8 doc with 7 internal newlines is ~71 chars.
# block_size must be >= that to generate full 8 lines reliably.
n_embd = 32
n_head = 4
n_layer = 1
block_size = 96

assert n_embd % n_head == 0
head_dim = n_embd // n_head

def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0.0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "lm_head": matrix(vocab_size, n_embd),
}

for i in range(n_layer):
    state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# -----------------------------------------------------------------------------
# 5) Model forward pass
# -----------------------------------------------------------------------------
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]

def softmax(logits):
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Attention
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])

        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [kk[hs:hs + head_dim] for kk in keys[li]]
            v_h = [vv[hs:hs + head_dim] for vv in values[li]]

            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / (head_dim ** 0.5)
                for t in range(len(k_h))
            ]
            attn_w = softmax(attn_logits)

            head_out = [
                sum(attn_w[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]

        # 2) MLP
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits

# -----------------------------------------------------------------------------
# 6) Adam optimizer
# -----------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

# -----------------------------------------------------------------------------
# 7) Training loop
# -----------------------------------------------------------------------------
num_steps = 2000  # bump a bit for better samples
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]

    n = min(block_size, len(tokens) - 1)
    if n <= 0:
        continue

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    loss = (1.0 / n) * sum(losses)
    loss.backward()

    lr_t = learning_rate * (1.0 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1.0 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1.0 - beta2) * (p.grad ** 2)
        m_hat = m[i] / (1.0 - beta1 ** (step + 1))
        v_hat = v[i] / (1.0 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0.0

    if (step + 1) % 50 == 0 or step == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# -----------------------------------------------------------------------------
# 8) Save checkpoint (so you can generate later without retraining)
# -----------------------------------------------------------------------------
model_state = {key: [[p.data for p in row] for row in mat] for key, mat in state_dict.items()}

with open("ascii_model.pkl", "wb") as f:
    pickle.dump(
        {
            "state_dict": model_state,
            "uchars": uchars,
            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "block_size": block_size,
        },
        f,
    )

print("✅ Model saved to ascii_model.pkl")

# -----------------------------------------------------------------------------
# 9) Inference: generate new 8x8 digit-like samples
# -----------------------------------------------------------------------------
def _sample_text(temperature=0.9, prompt=""):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    out = []

    # Force prompt tokens first (used for conditioned generation).
    for pos_id, ch in enumerate(prompt):
        if ch not in stoi:
            raise ValueError(f"Prompt char {ch!r} not present in vocabulary")
        _ = gpt(token_id, pos_id, keys, values)
        token_id = stoi[ch]

    start_pos = len(prompt)
    for pos_id in range(start_pos, block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs], k=1)[0]

        if token_id == BOS:
            break

        ch = uchars[token_id]
        out.append(ch)

        # Stop if the model emits a blank line (doc separator) => "\n\n"
        if len(out) >= 2 and out[-1] == "\n" and out[-2] == "\n":
            break

    return "".join(out).rstrip("\n")


def format_8x8(s):
    # Ignore optional conditioning header line if present.
    lines = s.splitlines()
    if lines and lines[0].startswith("@"):
        lines = lines[1:]
    return "\n".join(lines[:8])


def sample_conditioned(digit, temperature=0.9):
    if digit < 0 or digit > 9:
        raise ValueError("digit must be in range 0..9")
    return _sample_text(temperature=temperature, prompt=f"@{digit}\n")


temperature = 0.9
print("\n--- inference (generated 8x8 samples) ---")
for sample_idx in range(20):
    if conditioned_docs == len(docs):
        requested_digit = sample_idx % 10
        s = sample_conditioned(requested_digit, temperature=temperature)
        print(f"sample {sample_idx+1:2d} (requested digit={requested_digit}):\n{format_8x8(s)}\n")
    else:
        s = _sample_text(temperature=temperature)
        print(f"sample {sample_idx+1:2d}:\n{format_8x8(s)}\n")

