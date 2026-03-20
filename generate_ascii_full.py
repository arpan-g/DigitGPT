"""
generate_ascii_full.py (digits 8x8 visualizer + gallery)

Loads ascii_model.pkl (trained by your microgpt_digits.py), generates 8x8 digit-like
grids (chars in 0123456789ABCDEFG + newlines), renders them as PNG, and serves:

- Web gallery: http://127.0.0.1:8000/
- Optional single PNG endpoint: /png?t=0.9&seed=123&scale=24

Install:
  pip install pillow flask

Run:
  python generate_ascii_full.py --web
  python generate_ascii_full.py          # CLI interactive
"""

import argparse
import base64
import io
import math
import pickle
import random
from typing import List, Optional

from PIL import Image
from flask import Flask, Response, request, render_template_string

# -----------------------------
# Load checkpoint
# -----------------------------
CKPT_PATH = "ascii_model.pkl"
with open(CKPT_PATH, "rb") as f:
    ckpt = pickle.load(f)

required = ["state_dict", "uchars", "n_embd", "n_head", "n_layer", "block_size"]
missing = [k for k in required if k not in ckpt]
if missing:
    raise KeyError(
        f"Checkpoint {CKPT_PATH} missing keys: {missing}. "
        f"Re-save the model with these fields included."
    )
for tensor_name in ["wte", "wpe", "wre", "wce", "lm_head"]:
    if tensor_name not in ckpt["state_dict"]:
        raise KeyError(
            f"Checkpoint {CKPT_PATH} missing state tensor {tensor_name!r}. "
            "Re-train the model so 2D row/column embeddings are saved."
        )

state_dict = ckpt["state_dict"]
uchars = ckpt["uchars"]
vocab_size = ckpt.get("vocab_size", len(uchars) + 1)

n_embd = ckpt["n_embd"]
n_head = ckpt["n_head"]
n_layer = ckpt["n_layer"]
block_size = ckpt["block_size"]

BOS = len(uchars)
head_dim = n_embd // n_head
assert n_embd % n_head == 0
max_grid_rows = ckpt.get("max_grid_rows", 8)
max_grid_cols = ckpt.get("max_grid_cols", 8)
special_row_col = max(max_grid_rows, max_grid_cols)

def _to_float(x):
    return float(x.data) if hasattr(x, "data") else float(x)

for k, mat in list(state_dict.items()):
    state_dict[k] = [[_to_float(v) for v in row] for row in mat]

print(
    f"Loaded model: vocab_size={vocab_size}, n_embd={n_embd}, "
    f"n_head={n_head}, n_layer={n_layer}, block_size={block_size}"
)

# -----------------------------
# Digits encoding (0..16)
# -----------------------------
DIGIT_ALPH = "0123456789ABCDEFG"  # 17 chars => 0..16
CHAR_TO_VAL = {ch: i for i, ch in enumerate(DIGIT_ALPH)}

def normalize_to_8x8(text: str) -> List[str]:
    """
    Convert generated text to exactly 8 lines, each 8 chars.
    Unknown chars => '0'. Too short => padded with '0'. Too long => truncated.

    If text includes a conditioning header line like "@5", it is ignored.
    """
    lines = text.splitlines()
    if lines and lines[0].startswith("@"):
        lines = lines[1:]
    lines = lines[:8]

    out = []
    for line in lines:
        cleaned = "".join(ch if ch in CHAR_TO_VAL else "0" for ch in line)
        out.append(cleaned[:8].ljust(8, "0"))
    while len(out) < 8:
        out.append("0" * 8)
    return out

def grid_to_png_bytes(grid_lines: List[str], scale: int = 8) -> bytes:
    """
    grid_lines: 8 strings of length 8, chars in DIGIT_ALPH.
    Returns a scaled PNG (nearest-neighbor) as bytes.
    """
    img = Image.new("L", (8, 8))
    px = img.load()

    for y in range(8):
        for x in range(8):
            v = CHAR_TO_VAL.get(grid_lines[y][x], 0)  # 0..16
            gray = int((v / 16.0) * 255)              # 0..255
            px[x, y] = gray

    scale = max(1, int(scale))
    img = img.resize((8 * scale, 8 * scale), resample=Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return "data:image/png;base64," + b64

# -----------------------------
# Math helpers (float-only)
# -----------------------------
def linear(x, w):
    out = []
    for row in w:
        s = 0.0
        for wi, xi in zip(row, x):
            s += wi * xi
        out.append(s)
    return out

def softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]

def rmsnorm(x, eps=1e-5):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = 1.0 / math.sqrt(ms + eps)
    return [xi * scale for xi in x]

def relu_vec(x):
    return [xi if xi > 0.0 else 0.0 for xi in x]

def token_grid_position(token_history, pos_id):
    row = 0
    col = 0
    for i in range(pos_id):
        token = token_history[i]
        if token == BOS:
            continue
        ch = uchars[token]
        if ch == "\n":
            row += 1
            col = 0
            continue
        if ch == "@":
            continue
        if ch.isdigit() and i > 0 and token_history[i - 1] != BOS and uchars[token_history[i - 1]] == "@":
            continue
        if row < max_grid_rows and col < max_grid_cols:
            col += 1

    row = min(row, special_row_col)
    col = min(col, special_row_col)
    return row, col

# -----------------------------
# One-token forward
# -----------------------------
def gpt_step(token_id, pos_id, keys, values):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    row_id, col_id = token_grid_position(keys["token_history"], pos_id)
    row_emb = state_dict["wre"][row_id]
    col_emb = state_dict["wce"][col_id]
    x = [t + p + r + c for t, p, r, c in zip(tok_emb, pos_emb, row_emb, col_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # Attention
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])

        keys["layers"][li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h_all = [kk[hs:hs + head_dim] for kk in keys["layers"][li]]
            v_h_all = [vv[hs:hs + head_dim] for vv in values[li]]

            attn_logits = []
            for t in range(len(k_h_all)):
                dot = 0.0
                kt = k_h_all[t]
                for j in range(head_dim):
                    dot += q_h[j] * kt[j]
                attn_logits.append(dot / math.sqrt(head_dim))

            attn_w = softmax(attn_logits)

            head_out = []
            for j in range(head_dim):
                s = 0.0
                for t in range(len(v_h_all)):
                    s += attn_w[t] * v_h_all[t][j]
                head_out.append(s)

            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]

        # MLP
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = relu_vec(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits

# -----------------------------
# Sampling
# -----------------------------
def generate_raw(temperature: float = 0.9, max_len: Optional[int] = None, seed: Optional[int] = None, prompt: str = "") -> str:
    """
    Generate raw text (contains newlines) from BOS until stop condition.
    Optional prompt lets us condition generation (e.g. "@5\n").
    Uses a local RNG to avoid cross-request interference in web mode.
    """
    rng = random.Random(seed) if seed is not None else random

    if max_len is None:
        max_len = block_size
    max_len = min(int(max_len), block_size)

    keys = {"layers": [[] for _ in range(n_layer)], "token_history": []}
    values = [[] for _ in range(n_layer)]

    token_id = BOS
    out_chars = []

    temp = max(1e-6, float(temperature))
    newlines = 0

    # Force-feed prompt tokens first to condition the continuation.
    prompt_tokens = []
    for ch in prompt:
        if ch not in uchars:
            raise ValueError(f"Prompt char {ch!r} not found in model vocabulary")
        prompt_tokens.append(uchars.index(ch))

    pos_id = 0
    for forced_id in prompt_tokens:
        if pos_id >= max_len:
            return ""
        keys["token_history"].append(token_id)
        _ = gpt_step(token_id, pos_id, keys, values)
        token_id = forced_id
        pos_id += 1

    for pos_id in range(pos_id, max_len):
        keys["token_history"].append(token_id)
        logits = gpt_step(token_id, pos_id, keys, values)
        probs = softmax([z / temp for z in logits])
        token_id = rng.choices(range(vocab_size), weights=probs, k=1)[0]

        if token_id == BOS:
            break

        ch = uchars[token_id]
        out_chars.append(ch)

        # stop on doc boundary "\n\n"
        if len(out_chars) >= 2 and out_chars[-1] == "\n" and out_chars[-2] == "\n":
            break

        # heuristic: once we likely produced 8 lines, stop
        if ch == "\n":
            newlines += 1
            if newlines >= 8:
                break

    return "".join(out_chars).rstrip("\n")

def generate_digit_grid(temperature: float = 0.9, seed: Optional[int] = None, digit: Optional[int] = None) -> List[str]:
    prompt = ""
    if digit is not None:
        if digit < 0 or digit > 9:
            raise ValueError("digit must be in range 0..9")
        prompt = f"@{digit}\n"
    raw = generate_raw(temperature=temperature, max_len=block_size, seed=seed, prompt=prompt)
    return normalize_to_8x8(raw)

# -----------------------------
# Web gallery (Flask)
# -----------------------------
app = Flask(__name__)

HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>microGPT 8x8 Digits Gallery</title>
    <style>
      body { font-family: sans-serif; max-width: 1100px; margin: 20px auto; }
      form { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
      .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; margin-top: 16px; }
      .card { border: 1px solid #ddd; padding: 10px; border-radius: 8px; background: #fff; }
      img { border: 1px solid #eee; image-rendering: pixelated; width: 192px; height: 192px; }
      pre { margin: 8px 0 0; background: #f6f6f6; padding: 8px; font-size: 12px; line-height: 1.1; overflow: auto; }
      .meta { color: #555; font-size: 12px; margin-top: 6px; }
      .hint { color: #666; font-size: 12px; margin-top: 4px; }
    </style>
  </head>
  <body>
    <h2>microGPT 8×8 digit generator (gallery)</h2>

    <form method="GET" action="/">
      <label>Count: <input name="n" value="{{n}}" size="4"/></label>
      <label>Temperature: <input name="t" value="{{t}}" size="6"/></label>
      <label>Seed (optional): <input name="seed" value="{{seed}}" size="10"/></label>
      <label>Scale: <input name="scale" value="{{scale}}" size="4"/></label>
      <label>Digit (optional 0-9): <input name="d" value="{{digit}}" size="4"/></label>
      <button type="submit">Generate gallery</button>
    </form>
    <div class="hint">Tip: set seed=123 to make the gallery repeatable; raise temperature for more variety.</div>
    <div class="hint">Set digit=0..9 to condition the generator on a specific target digit.</div>

    <div class="grid">
      {% for item in items %}
        <div class="card">
        <img src="{{item.data_url}}" style="width:auto;height:auto;" />
          <div class="meta">seed={{item.seed}}</div>
          <pre>{{item.grid_text}}</pre>
        </div>
      {% endfor %}
    </div>
  </body>
</html>
"""

@app.get("/")
def index():
    t = float(request.args.get("t", "0.9"))
    n = int(request.args.get("n", "30"))
    n = max(1, min(n, 200))  # safety cap

    scale = int(request.args.get("scale", "24"))
    scale = max(1, min(scale, 64))

    seed_str = request.args.get("seed", "").strip()
    base_seed = int(seed_str) if seed_str != "" else None

    digit_str = request.args.get("d", "").strip()
    digit = int(digit_str) if digit_str != "" else None
    if digit is not None:
        digit = max(0, min(9, digit))

    items = []
    for i in range(n):
        s = (base_seed + i) if base_seed is not None else None
        grid = generate_digit_grid(temperature=t, seed=s, digit=digit)
        png = grid_to_png_bytes(grid, scale=scale)
        items.append(
            {
                "seed": s,
                "digit": digit,
                "grid_text": "\n".join(grid),
                "data_url": png_bytes_to_data_url(png),
            }
        )

    return render_template_string(
        HTML,
        t=str(t),
        n=str(n),
        seed=seed_str,
        scale=str(scale),
        digit=digit_str,
        items=items,
    )

@app.get("/png")
def png_single():
    t = float(request.args.get("t", "0.9"))
    scale = int(request.args.get("scale", "8"))
    scale = max(1, min(scale, 64))

    seed_str = request.args.get("seed", "").strip()
    seed = int(seed_str) if seed_str != "" else None

    digit_str = request.args.get("d", "").strip()
    digit = int(digit_str) if digit_str != "" else None
    if digit is not None:
        digit = max(0, min(9, digit))

    grid = generate_digit_grid(temperature=t, seed=seed, digit=digit)
    png = grid_to_png_bytes(grid, scale=scale)
    return Response(png, mimetype="image/png")

def run_web(host: str, port: int):
    app.run(host=host, port=port, debug=True)

# -----------------------------
# CLI interactive mode
# -----------------------------
def cli():
    print("\nCommands: g 10 | d 5 | t 0.9 | seed 123 | png on/off | scale 24 | q\n")
    temperature = 0.9
    seed = None
    autosave_png = False
    scale = 24
    idx = 0
    digit = None

    while True:
        cmd = input("> ").strip()
        if not cmd:
            continue
        if cmd.lower() == "q":
            break

        if cmd.startswith("t "):
            temperature = float(cmd.split()[1])
            print(f"temperature={temperature}")
            continue

        if cmd.startswith("seed "):
            seed = int(cmd.split()[1])
            print(f"seed={seed}")
            continue

        if cmd.startswith("d "):
            d = int(cmd.split()[1])
            if d < 0 or d > 9:
                print("digit must be in range 0..9")
                continue
            digit = d
            print(f"digit={digit}")
            continue

        if cmd.startswith("scale "):
            scale = int(cmd.split()[1])
            print(f"scale={scale}")
            continue

        if cmd.startswith("png "):
            v = cmd.split()[1].lower()
            autosave_png = (v == "on")
            print(f"autosave_png={autosave_png}")
            continue

        if cmd.startswith("g"):
            parts = cmd.split()
            n = int(parts[1]) if len(parts) > 1 else 5

            for i in range(n):
                s = (seed + i) if seed is not None else None
                grid = generate_digit_grid(temperature=temperature, seed=s, digit=digit)
                print(f"\nSample {idx} (seed={s}, digit={digit}):")
                print("\n".join(grid))

                if autosave_png:
                    out_path = f"digit_{idx:04d}.png"
                    with open(out_path, "wb") as f:
                        f.write(grid_to_png_bytes(grid, scale=scale))
                    print(f"Saved {out_path}")

                idx += 1
            print()
            continue

        print("Unknown. Use: g 10 | t 0.9 | seed 123 | png on/off | scale 24 | q")

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--web", action="store_true", help="Run web gallery UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    if args.web:
        run_web(args.host, args.port)
    else:
        cli()
