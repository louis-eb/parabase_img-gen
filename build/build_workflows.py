import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from wfbuilder import *

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)

# ---------------- shared prompt defaults (English: the models read English best) ----------------
MOTIV = ("Close-up architectural photograph of a building fragment, exactly as drawn in the sketch: "
         "a deep window reveal cut into a thick wall, a slender lintel above, the edge of the opening "
         "seen slightly from below. Only a part of the building is visible, cropped tightly, no sky line, "
         "no full facade.")
MATERIAL = ("The wall is board-marked in-situ concrete with a fine timber grain imprint, small air pores, "
            "subtle efflorescence and darker weathering streaks below the lintel. The lintel is dark "
            "patinated steel with a faint rust bloom. The window is a deep frameless glass pane reflecting "
            "a soft grey sky.")
STIMMUNG = ("Shot on a medium format camera, 80mm lens, f/5.6, eye-level tripod shot. Soft overcast "
            "late-afternoon light grazing across the surface, revealing every texture, deep soft shadows "
            "inside the reveal. Muted, desaturated palette of warm greys and dark umber, quiet, contemplative, "
            "almost still-life atmosphere, light haze, subtle film grain. Precise physical material detail, "
            "photorealistic architectural photography.")

GREEN, BLUE, BROWN, PURPLE, GREY = "#3f6e3f", "#3f789e", "#8a6a3a", "#6b4f8a", "#555555"


# =====================================================================================
# A  —  Skizze -> Stimmung   (Z-Image Turbo + Fun ControlNet Union 2.1 / Scribble)
# =====================================================================================
def build_A():
    wf = WF()
    # --- notes
    wf.note((-1000, -40), """## A · Skizze → Stimmung
**Z-Image Turbo + Fun ControlNet Union 2.1 (Scribble)**

So arbeitest du damit:
1. Skizze im Node **Skizze** hochladen (Bleistift, Fineliner, Screenshot – egal). Hochformat/Querformat wird übernommen.
2. Die drei grünen Text-Blöcke anpassen: **Motiv**, **Material**, **Stimmung & Kamera**. Sie werden automatisch zu einem Prompt zusammengesetzt.
3. **Run** – du bekommst 4 Varianten (Batch-Grösse im Node *Bildgrösse & Anzahl*).
4. Das beste Bild per Rechtsklick → *Save Image* sichern und in Workflow **C · Upscale** laden.

**Wichtigste Regler**
- **Kontrollbild prüfen!** Es muss **weisse Linien auf reinem Schwarz** zeigen. Graue Flächen oder Papierstruktur → das Modell malt die Skizze ab statt ein Foto zu machen.
- **Linien-Schwelle** (0.4–0.7): Fehlen Linien → tiefer. Papierflecken/Schatten sichtbar → höher. **Linien verdicken**: 0–3.
- **ControlNet-Stärke** (Node *Skizze → Geometrie*): 0.6 = lockere Interpretation, 0.9 = sehr nah an der Skizze. Start: 0.7.
- **end_percent** (unter *advanced*): 0.7–0.8 lässt dem Modell in den letzten Schritten Freiheit für Materialdetails.
- **Schalter**: *false* = Handskizze als Scribble (oben). *true* = Canny-Kanten (für Screenshots aus dem 3D-Modell).
- **Seed** auf *fixed* stellen, wenn du an einem Bild nur den Prompt verfeinern willst.

Steps 8 · CFG 1 · res_multistep/simple · Shift 3 sind auf das 8-Step-Modell abgestimmt – nicht verändern.""",
            size=(560, 640), color="#223", bgcolor="#335")

    # --- loaders
    wf.group("Modelle", -1000, 640, 440, 470, GREY)
    m = unet(wf, (-980, 690), "z_image_turbo_bf16.safetensors")
    c = clip(wf, (-980, 800), "qwen_3_4b.safetensors", "lumina2")
    v = vae(wf, (-980, 930), "ae.safetensors")
    mp = wf.node("ModelPatchLoader", (-980, 1010), ["Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"],
                 outputs=[("MODEL_PATCH", "MODEL_PATCH")], size=(400, 58))

    # --- sketch prep
    wf.group("1 · Skizze vorbereiten", -400, -40, 1060, 760, BLUE)
    img = load_image(wf, (-380, 10), "skizze_beispiel.png", title="Skizze")
    sc = scale_total(wf, (-40, 10), 1.5)
    sc["title"] = "Skizze auf ~1.5 MP skalieren"
    # Skizze -> saubere Scribble-Linien: weiss auf reinem Schwarz, Papiergrau und Schatten weg.
    # Blau-Kanal: Bleistift UND orange/rote Marker erscheinen dort dunkel.
    bl = wf.node("ImageBlur", (-40, 150), [1, 1.0], inputs=[("image", "IMAGE")], outputs=[("IMAGE", "IMAGE")],
                 size=(300, 82), title="Papierrauschen glätten")
    i2m = wf.node("ImageToMask", (-40, 260), ["blue"], inputs=[("image", "IMAGE")], outputs=[("MASK", "MASK")],
                  size=(300, 58), title="Helligkeit (Blau-Kanal)")
    im = wf.node("InvertMask", (-40, 350), [], inputs=[("mask", "MASK")], outputs=[("MASK", "MASK")],
                 size=(300, 30), title="Invertieren (Linien weiss)")
    th = wf.node("ThresholdMask", (-40, 420), [0.55], inputs=[("mask", "MASK")], outputs=[("MASK", "MASK")],
                 size=(300, 58), title="Linien-Schwelle (0.4–0.7)")
    gr = wf.node("GrowMask", (-40, 510), [1, True], inputs=[("mask", "MASK")], outputs=[("MASK", "MASK")],
                 size=(300, 82), title="Linien verdicken (expand)")
    m2i = wf.node("MaskToImage", (-40, 620), [], inputs=[("mask", "MASK")], outputs=[("IMAGE", "IMAGE")],
                  size=(300, 30))
    can = wf.node("Canny", (300, 430), [0.15, 0.35], inputs=[("image", "IMAGE")], outputs=[("IMAGE", "IMAGE")],
                  size=(340, 82), title="Canny (für 3D-Screenshots)")
    sw = wf.node("ComfySwitchNode", (300, 540), [False],
                 inputs=[("on_false", "IMAGE"), ("on_true", "IMAGE")], outputs=[("output", "IMAGE")],
                 size=(340, 82), title="Skizze/Scribble (false) / Canny (true)")
    pv = preview(wf, (300, 10), size=(340, 400), title="Kontrollbild – muss weisse Linien auf Schwarz zeigen")
    wf.link(img, 0, sc, "image"); wf.link(sc, 0, bl, "image"); wf.link(bl, 0, i2m, "image")
    wf.link(i2m, 0, im, "mask"); wf.link(im, 0, th, "mask"); wf.link(th, 0, gr, "mask"); wf.link(gr, 0, m2i, "mask")
    wf.link(sc, 0, can, "image")
    wf.link(m2i, 0, sw, "on_false"); wf.link(can, 0, sw, "on_true"); wf.link(sw, 0, pv, "images")

    # --- prompt
    wf.group("2 · Prompt", -400, 740, 1060, 640, GREEN)
    t1 = text_block(wf, (-380, 790), MOTIV, "Motiv / Fragment")
    t2 = text_block(wf, (-380, 980), MATERIAL, "Material")
    t3 = text_block(wf, (-380, 1170), STIMMUNG, "Stimmung & Kamera")
    k1 = concat(wf, (110, 860)); k2 = concat(wf, (110, 1060))
    wf.link(t1, 0, k1, "string_a"); wf.link(t2, 0, k1, "string_b")
    wf.link(k1, 0, k2, "string_a"); wf.link(t3, 0, k2, "string_b")
    pos_ = clip_encode(wf, (400, 900), "Prompt (positiv)")
    neg = zero_out(wf, (400, 1060))
    wf.link(c, 0, pos_, "clip"); wf.link(k2, 0, pos_, "text"); wf.link(pos_, 0, neg, "conditioning")

    # --- generate
    wf.group("3 · Generieren", 700, -40, 980, 1420, PURPLE)
    cn = wf.node("QwenImageDiffsynthControlnet", (720, 10), [0.7, 0.0, 0.8],
                 inputs=[("model", "MODEL"), ("model_patch", "MODEL_PATCH"), ("vae", "VAE"), ("image", "IMAGE"),
                         ("mask", "MASK")],
                 outputs=[("MODEL", "MODEL")], size=(320, 160), title="Skizze → Geometrie (ControlNet)")
    wf.link(m, 0, cn, "model"); wf.link(mp, 0, cn, "model_patch"); wf.link(v, 0, cn, "vae"); wf.link(sw, 0, cn, "image")
    ms = wf.node("ModelSamplingAuraFlow", (720, 210), [3], inputs=[("model", "MODEL")], outputs=[("MODEL", "MODEL")],
                 size=(320, 58))
    wf.link(cn, 0, ms, "model")
    gs = wf.node("GetImageSize", (720, 310), [], inputs=[("image", "IMAGE")],
                 outputs=[("width", "INT"), ("height", "INT"), ("batch_size", "INT")], size=(220, 66))
    wf.link(sc, 0, gs, "image")
    lat = wf.node("EmptySD3LatentImage", (720, 420), [1024, 1024, 4],
                  inputs=[("width", "INT", "width"), ("height", "INT", "height")],
                  outputs=[("LATENT", "LATENT")], size=(320, 106), title="Bildgrösse & Anzahl (batch_size)")
    wf.link(gs, 0, lat, "width"); wf.link(gs, 1, lat, "height")
    ks = ksampler(wf, (720, 580), 20261019, 8, 1, "res_multistep", "simple", 1.0)
    wf.link(ms, 0, ks, "model"); wf.link(pos_, 0, ks, "positive"); wf.link(neg, 0, ks, "negative"); wf.link(lat, 0, ks, "latent_image")
    dec = vae_decode(wf, (720, 880)); wf.link(ks, 0, dec, "samples"); wf.link(v, 0, dec, "vae")
    sv = save_image(wf, (1080, 10), "A_skizze/variante", size=(580, 1340))
    wf.link(dec, 0, sv, "images")
    return wf


# =====================================================================================
# B  —  Skizze + Materialbilder   (FLUX.3 Image Klein, multi-reference)
# =====================================================================================
PROMPT_B = ("Create a photorealistic close-up architectural photograph of the building fragment drawn in "
            "<ref_image_0>. Keep the exact geometry, proportions, edges, openings and camera angle of the sketch; "
            "only a part of the building is visible, tightly cropped. Render the main wall surfaces in the material "
            "shown in <ref_image_1>, matching its colour, texture, grain, joint pattern and scale. Render the frames, "
            "edges and lintel in the material shown in <ref_image_2>. Medium format camera, 80mm lens, soft overcast "
            "late-afternoon light grazing across the surfaces, deep soft shadows inside the reveal, muted desaturated "
            "palette, quiet contemplative atmosphere, light haze, subtle film grain. Precise physical material detail. "
            "No pencil lines, paper texture or drawing marks remain.")


def build_B():
    wf = WF()
    wf.note((-1000, -40), """## B · Skizze + Materialbilder
**FLUX.3 Image Klein (Multi-Referenz)**

Hier steuerst du Materialien mit **Fotos** statt nur mit Text.

1. **Referenz 0 = Skizze** (bestimmt Geometrie, Ausschnitt und Bildformat).
2. **Referenz 1 / 2 = Materialfotos** – z. B. Beton-Musterplatte, Holz, Naturstein, Metall. Am besten frontal, gleichmässig belichtet, nur das Material im Bild.
3. Im Prompt sprichst du die Bilder mit `<ref_image_0>`, `<ref_image_1>`, `<ref_image_2>` … an – in der Reihenfolge, in der sie am Node *Referenzen* hängen.
4. **Optional Stimmungsreferenz**: Den Node *Stimmung (optional)* mit dem freien Eingang `images.image3` am Node *Referenzen* verbinden und im Prompt z. B. schreiben: *"Match the light, colour grading and atmosphere of <ref_image_3>, but not its content."*
5. **Run** → 4 Varianten. Bestes Bild in **C · Upscale** weiterverarbeiten.

**Material entfernen**: Verbindung zum Node *Referenzen* lösen und den Satz im Prompt löschen. **Mehr Materialien**: einfach weitere Load-Image-Nodes an die freien Eingänge hängen (bis 10 Referenzen).

**Tipps**
- Prompt als vollständige Bildbeschreibung schreiben, nicht als Befehl.
- Wenn die Skizze zu frei interpretiert wird: im Prompt betonen *"keep the exact geometry … of <ref_image_0>"* und die Skizze sauberer/kontrastreicher machen.
- `megapixels` am Node *Referenzen* = Ausgabegrösse (1.5 ≈ 1500×1000). Höher = mehr Materialdetail, aber langsamer.

Steps 16 · CFG 1 · euler/simple (destilliertes Modell).""",
            size=(560, 760), color="#223", bgcolor="#335")

    wf.group("Modelle", -1000, 760, 440, 330, GREY)
    m = unet(wf, (-980, 810), "flux3-image-klein-bf16.safetensors")
    c = clip(wf, (-980, 910), "qwen3vl_4b_int8_convrot.safetensors", "flux3")
    v = vae(wf, (-980, 1030), "flux2-vae.safetensors")

    wf.group("1 · Referenzbilder", -400, -40, 1400, 470, BLUE)
    sk = load_image(wf, (-380, 10), "skizze_beispiel.png", title="Referenz 0 · Skizze")
    m1 = load_image(wf, (-40, 10), "material_1.png", title="Referenz 1 · Material Wand")
    m2 = load_image(wf, (300, 10), "material_2.png", title="Referenz 2 · Material Rahmen/Kante")
    st = load_image(wf, (640, 10), "stimmung.png", title="Stimmung (optional, nicht verbunden)")

    wf.group("2 · Prompt", -400, 460, 1400, 430, GREEN)
    tp = text_block(wf, (-380, 510), PROMPT_B, "Prompt (Bildbeschreibung mit <ref_image_N>)", size=(760, 340))
    pe = clip_encode(wf, (420, 520), "Prompt (positiv)")
    ng = zero_out(wf, (420, 660))
    wf.link(c, 0, pe, "clip"); wf.link(tp, 0, pe, "text"); wf.link(pe, 0, ng, "conditioning")

    wf.group("3 · Generieren", 1040, -40, 1100, 1300, PURPLE)
    ref = wf.node("ReferenceMultiLatent", (1060, 10), ["resize", 1.5],
                  inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("vae", "VAE"),
                          ("images.image0", "IMAGE"), ("images.image1", "IMAGE"), ("images.image2", "IMAGE"),
                          ("images.image3", "IMAGE")],
                  outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("empty_latent", "LATENT")],
                  size=(340, 230), title="Referenzen")
    wf.link(pe, 0, ref, "positive"); wf.link(ng, 0, ref, "negative"); wf.link(v, 0, ref, "vae")
    wf.link(sk, 0, ref, "images.image0"); wf.link(m1, 0, ref, "images.image1"); wf.link(m2, 0, ref, "images.image2")
    rb = wf.node("RepeatLatentBatch", (1060, 290), [4], inputs=[("samples", "LATENT")], outputs=[("LATENT", "LATENT")],
                 size=(340, 58), title="Anzahl Varianten")
    wf.link(ref, 2, rb, "samples")
    ks = ksampler(wf, (1060, 400), 20261019, 16, 1, "euler", "simple", 1.0)
    wf.link(m, 0, ks, "model"); wf.link(ref, 0, ks, "positive"); wf.link(ref, 1, ks, "negative"); wf.link(rb, 0, ks, "latent_image")
    dec = vae_decode(wf, (1060, 700)); wf.link(ks, 0, dec, "samples"); wf.link(v, 0, dec, "vae")
    sv = save_image(wf, (1440, 10), "B_material/variante", size=(680, 1220))
    wf.link(dec, 0, sv, "images")
    return wf


# =====================================================================================
# C  —  Upscale   (Z-Image Detail-Pass + SeedVR2 7B)
# =====================================================================================
def build_C():
    wf = WF()
    wf.note((-1000, -40), """## C · Upscale eines ausgewählten Bildes
**Stufe 1 – Detail-Pass (Z-Image Turbo):** Bild wird 4× hochgerechnet, auf ~4 MP gebracht und mit wenig Denoise neu „durchgezeichnet“. Das bringt echte neue Materialdetails (Poren, Maserung, Patina).
**Stufe 2 – SeedVR2 7B:** macht daraus ein scharfes, sauberes Bild in hoher Auflösung (Standard ×2 → ca. 16 MP, z. B. 4900×3260).

So arbeitest du damit:
1. Ausgewähltes Bild aus A oder B im Node **Ausgewähltes Bild** hochladen (oder aus dem Output-Ordner per Drag & Drop).
2. Den **gleichen Prompt** wie bei der Generierung in den grünen Block kopieren (für B ohne die `<ref_image>`-Bezüge reicht eine Kurzfassung).
3. **Run**. Mit dem Vergleichs-Node siehst du vorher/nachher.

**Regler**
- **Detail-Denoise** (KSampler Stufe 1): 0.25 = sehr nah am Original, 0.40 = mehr neue Details, aber kleine Formänderungen möglich. **0 = Stufe 1 aus.**
- **Zielgrösse Stufe 1** (megapixels): 4 ist ein guter Wert. Bei kleinen Ausgangsbildern nicht über 6.
- **SeedVR2-Faktor** (multiplier): 2 = ca. 16 MP. Bei Speicherfehler (out of memory) auf 1.5 reduzieren.
- Farbkorrektur *lab* hält die Farben des Originals.""",
            size=(560, 640), color="#223", bgcolor="#335")

    wf.group("Modelle", -1000, 640, 440, 560, GREY)
    m = unet(wf, (-980, 690), "z_image_turbo_bf16.safetensors")
    c = clip(wf, (-980, 790), "qwen_3_4b.safetensors", "lumina2")
    v = vae(wf, (-980, 910), "ae.safetensors")
    up = wf.node("UpscaleModelLoader", (-980, 990), ["RealESRGAN_x4plus.safetensors"],
                 outputs=[("UPSCALE_MODEL", "UPSCALE_MODEL")], size=(400, 58))
    sm = unet(wf, (-980, 1070), "seedvr2_7b_int8_convrot.safetensors"); sm["title"] = "SeedVR2 Modell"
    sv_vae = vae(wf, (-980, 1150), "seedvr2_ema_vae_fp16.safetensors"); sv_vae["title"] = "SeedVR2 VAE"

    wf.group("0 · Eingang & Prompt", -400, -40, 760, 1000, BLUE)
    img = load_image(wf, (-380, 10), "ausgewaehlt.png", title="Ausgewähltes Bild")
    tp = text_block(wf, (-380, 430), MOTIV + " " + MATERIAL + " " + STIMMUNG,
                    "Prompt (gleicher wie bei der Generierung)", size=(700, 300))
    pe = clip_encode(wf, (-40, 760), "Prompt (positiv)")
    ng = zero_out(wf, (-40, 880))
    wf.link(c, 0, pe, "clip"); wf.link(tp, 0, pe, "text"); wf.link(pe, 0, ng, "conditioning")

    wf.group("1 · Detail-Pass (Z-Image)", 400, -40, 400, 1000, GREEN)
    ups = wf.node("ImageUpscaleWithModel", (420, 10), [], inputs=[("upscale_model", "UPSCALE_MODEL"), ("image", "IMAGE")],
                  outputs=[("IMAGE", "IMAGE")], size=(340, 46))
    wf.link(up, 0, ups, "upscale_model"); wf.link(img, 0, ups, "image")
    sc = scale_total(wf, (420, 100), 4.0); sc["title"] = "Zielgrösse Stufe 1 (megapixels)"
    wf.link(ups, 0, sc, "image")
    enc = vae_encode(wf, (420, 250)); wf.link(sc, 0, enc, "pixels"); wf.link(v, 0, enc, "vae")
    ms = wf.node("ModelSamplingAuraFlow", (420, 330), [3], inputs=[("model", "MODEL")], outputs=[("MODEL", "MODEL")],
                 size=(340, 58))
    wf.link(m, 0, ms, "model")
    ks = ksampler(wf, (420, 430), 20261019, 5, 1, "dpmpp_2m_sde", "beta", 0.3, control="fixed")
    ks["title"] = "KSampler Stufe 1 (Detail-Denoise)"
    wf.link(ms, 0, ks, "model"); wf.link(pe, 0, ks, "positive"); wf.link(ng, 0, ks, "negative"); wf.link(enc, 0, ks, "latent_image")
    dec = vae_decode(wf, (420, 730)); wf.link(ks, 0, dec, "samples"); wf.link(v, 0, dec, "vae")

    wf.group("2 · SeedVR2 Upscale", 840, -40, 400, 1000, PURPLE)
    rs = wf.node("ResizeImageMaskNode", (860, 10), ["scale by multiplier", 2.0, "lanczos"],
                 inputs=[("input", "IMAGE,MASK")], outputs=[("resized", "IMAGE")], size=(340, 106),
                 title="SeedVR2-Faktor (multiplier)")
    wf.link(dec, 0, rs, "input")
    pre = wf.node("SeedVR2Preprocess", (860, 160), [], inputs=[("resized_images", "IMAGE")], outputs=[("images", "IMAGE")],
                  size=(340, 30))
    wf.link(rs, 0, pre, "resized_images")
    et = wf.node("VAEEncodeTiled", (860, 230), [512, 128, 4096, 8], inputs=[("pixels", "IMAGE"), ("vae", "VAE")],
                 outputs=[("LATENT", "LATENT")], size=(340, 150))
    wf.link(pre, 0, et, "pixels"); wf.link(sv_vae, 0, et, "vae")
    cond = wf.node("SeedVR2Conditioning", (860, 420), [], inputs=[("model", "MODEL"), ("vae_conditioning", "LATENT")],
                   outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")], size=(340, 50))
    wf.link(sm, 0, cond, "model"); wf.link(et, 0, cond, "vae_conditioning")
    ks2 = ksampler(wf, (860, 510), 20261019, 1, 1, "euler", "simple", 1.0, control="fixed")
    ks2["title"] = "KSampler SeedVR2 (nicht ändern)"
    wf.link(sm, 0, ks2, "model"); wf.link(cond, 0, ks2, "positive"); wf.link(cond, 1, ks2, "negative"); wf.link(et, 0, ks2, "latent_image")
    dt = wf.node("VAEDecodeTiled", (860, 800), [512, 128, 4096, 8], inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")], size=(340, 150))
    wf.link(ks2, 0, dt, "samples"); wf.link(sv_vae, 0, dt, "vae")
    post = wf.node("SeedVR2PostProcessing", (860, 990), ["lab"],
                   inputs=[("images", "IMAGE"), ("original_resized_images", "IMAGE")],
                   outputs=[("images", "IMAGE")], size=(340, 82))
    wf.link(dt, 0, post, "images"); wf.link(rs, 0, post, "original_resized_images")

    wf.group("3 · Ergebnis", 1280, -40, 1300, 1180, BROWN)
    sv = save_image(wf, (1300, 10), "C_upscale/final", size=(620, 1100))
    wf.link(post, 0, sv, "images")
    cmp_ = wf.node("ImageCompare", (1940, 10), [], inputs=[("image_a", "IMAGE"), ("image_b", "IMAGE")], size=(620, 700),
                   title="Vorher / Nachher")
    wf.link(img, 0, cmp_, "image_a"); wf.link(post, 0, cmp_, "image_b")
    pv1 = preview(wf, (1940, 740), size=(620, 380), title="Zwischenstand nach Stufe 1")
    wf.link(dec, 0, pv1, "images")
    return wf


for name, fn in [("A_Skizze_zu_Stimmung.json", build_A), ("B_Skizze_plus_Materialbilder.json", build_B),
                 ("C_Upscale.json", build_C)]:
    fn().save(os.path.join(OUT, name))
    print("wrote", name)
