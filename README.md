# DigitGPT

Install dependencies:

```bash
pip install pillow flask scikit-learn
```

## 1) Build conditioned dataset

```bash
python dataset.py
```

This writes `ascii_dataset.txt` where each training document starts with a label header:

```
@5
<8x8 digit image rows>
```

So training learns to generate an image conditioned on a requested digit.

## 2) Train model

```bash
python microgpt.py
```

This creates `ascii_model.pkl`.

The trainer now uses both:
- 1D token-position embeddings, and
- 2D row/column embeddings for the 8x8 grid,

so the model can distinguish pixels that are on the same row, same column, or at different image locations more explicitly.

## 3) Generate images

### Web app

```bash
python generate_ascii_full.py --web
```

Open:

`http://127.0.0.1:8000/?n=10&t=0.9&seed=123&scale=24&d=5`

Query params:
- `n`: number of generated images (configurable, e.g. 10)
- `d`: optional requested digit (0..9)
- `t`: temperature
- `seed`: optional deterministic seed
- `scale`: output upscaling

### CLI

```bash
python generate_ascii_full.py
```

Commands:
- `g 10` generate 10 samples
- `d 5` condition on digit 5 (0..9)
- `t 0.9` set temperature
- `seed 123` set deterministic seed
- `png on/off` autosave PNG files
- `scale 24` PNG upscale factor
- `q` quit
