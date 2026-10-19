"""Example inputs so the workflows run on first launch: a pencil-style sketch of a facade fragment
and two procedural material swatches (board-marked concrete, patinated steel)."""
import sys, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
rng = random.Random(7)
np.random.seed(7)

# ---------------- sketch: deep window reveal, seen slightly from below ----------------
W, H = 1200, 1500
img = Image.new("L", (W, H), 246)
# paper grain
paper = (np.random.normal(0, 3.0, (H, W))).astype(np.int16)
img = Image.fromarray(np.clip(np.array(img, np.int16) + paper, 0, 255).astype(np.uint8))
d = ImageDraw.Draw(img)

def pencil(p, q, w=3, passes=2, jitter=2.5, tone=55):
    for _ in range(passes):
        a = (p[0] + rng.uniform(-jitter, jitter), p[1] + rng.uniform(-jitter, jitter))
        b = (q[0] + rng.uniform(-jitter, jitter), q[1] + rng.uniform(-jitter, jitter))
        # overshoot the ends a little, like a hand sketch
        dx, dy = b[0] - a[0], b[1] - a[1]
        a = (a[0] - dx * 0.02, a[1] - dy * 0.02); b = (b[0] + dx * 0.02, b[1] + dy * 0.02)
        d.line([a, b], fill=tone + rng.randint(-15, 15), width=w)

def poly(pts, **k):
    for i in range(len(pts) - 1):
        pencil(pts[i], pts[i + 1], **k)

# outer opening in the wall (slight 2-point perspective, from below)
o = [(250, 380), (930, 330), (950, 1180), (230, 1150)]
poly(o + [o[0]], w=4, passes=3)
# inner glass line (deep reveal on left + top)
i_ = [(400, 470), (900, 440), (915, 1120), (385, 1105)]
poly(i_ + [i_[0]], w=3)
for a, b in zip(o, i_):
    pencil(a, b, w=3)
# lintel: thin steel plate above the opening
poly([(180, 300), (1010, 245), (1012, 275), (182, 330), (180, 300)], w=4, passes=3)
# board-marked joints on the wall (horizontal, converging slightly)
for k, y in enumerate(range(140, 1450, 95)):
    if 300 < y < 1190:
        pencil((20, y + 8), (225, y), w=2, tone=120, passes=1)
        pencil((955, y - 6), (1180, y - 16), w=2, tone=120, passes=1)
    else:
        pencil((20, y + 8), (1180, y - 18), w=2, tone=120, passes=1)
# hatching inside the reveal (shadow)
for t in np.linspace(0, 1, 26):
    x0 = 250 + (230 - 250) * t; y0 = 380 + (1150 - 380) * t
    x1 = 400 + (385 - 400) * t; y1 = 470 + (1105 - 470) * t
    pencil((x0 + 6, y0 + 10), (x1 - 4, y1 - 2), w=2, tone=140, passes=1, jitter=1.5)
for t in np.linspace(0, 1, 18):
    x0 = 250 + (930 - 250) * t; y0 = 380 + (330 - 380) * t
    x1 = 400 + (900 - 400) * t; y1 = 470 + (440 - 470) * t
    pencil((x0 + 4, y0 + 8), (x1, y1 - 4), w=2, tone=150, passes=1, jitter=1.5)
img = img.filter(ImageFilter.GaussianBlur(0.6))
img.convert("RGB").save(os.path.join(OUT, "skizze_beispiel.png"))

# ---------------- material 1: board-marked concrete ----------------
S = 1024
def fractal(size, octaves=6, persistence=0.55):
    out = np.zeros((size, size)); amp = 1.0; tot = 0
    for o in range(octaves):
        n = 4 * 2 ** o
        g = np.random.rand(n + 1, n + 1)
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        out += amp * (np.asarray(im) / 255.0); tot += amp; amp *= persistence
    return out / tot
base = fractal(S)
conc = 150 + (base - 0.5) * 70
# timber grain stripes per board
for b in range(0, S, 128):
    yy = np.arange(b, min(b + 128, S))
    grain = np.sin((np.arange(S)[None, :] / rng.uniform(9, 16)) + np.cumsum(np.random.normal(0, 0.15, (len(yy), S)), axis=1) * 0.2)
    conc[yy, :] += grain * 6 + rng.uniform(-8, 8)
    conc[min(b, S - 1):min(b + 3, S), :] -= 28  # board joint
# pores
for _ in range(1400):
    x, y, r = rng.randrange(S), rng.randrange(S), rng.uniform(0.8, 2.8)
    yy, xx = np.ogrid[-4:5, -4:5]
    m = (xx ** 2 + yy ** 2) <= r * r
    ys, xs = slice(max(0, y - 4), min(S, y + 5)), slice(max(0, x - 4), min(S, x + 5))
    conc[ys, xs][m[:conc[ys, xs].shape[0], :conc[ys, xs].shape[1]]] -= 45
conc = np.clip(conc, 0, 255)
rgb = np.stack([conc * 1.0, conc * 0.985, conc * 0.95], -1)
Image.fromarray(rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)).save(os.path.join(OUT, "material_1.png"))

# ---------------- material 2: dark patinated steel ----------------
st = fractal(S, 7, 0.6)
brushed = np.asarray(Image.fromarray((np.random.rand(S, S) * 255).astype(np.uint8)).resize((S // 64, S), Image.BICUBIC).resize((S, S), Image.BICUBIC)) / 255.0
v = 48 + (st - 0.5) * 40 + (brushed - 0.5) * 14
rust = np.clip((fractal(S, 6, 0.6) - 0.6) * 3, 0, 1) ** 1.5
r = v + rust * 38; g = v * 0.97 + rust * 18; bl = v * 0.95 + rust * 6
Image.fromarray(np.clip(np.stack([r, g, bl], -1), 0, 255).astype(np.uint8)).save(os.path.join(OUT, "material_2.png"))

# ---------------- placeholders ----------------
# neutral grey "mood" placeholder (node is not connected by default)
Image.new("RGB", (512, 512), (128, 128, 128)).save(os.path.join(OUT, "stimmung.png"))
# example for workflow C: a copy of material_1 so the graph validates before a real image is chosen
Image.open(os.path.join(OUT, "material_1.png")).resize((768, 768)).save(os.path.join(OUT, "ausgewaehlt.png"))
print("examples written to", OUT)
