"""Build ComfyUI API graphs for FLUX.3 Image Klein (text-to-image or with reference images),
optionally with our own LoRAs. Used to write 'comfy' jobs for remote/agent.py.

    from klein_graph import graph
    g = graph("prbstyle, architectural photograph: ...", loras=[("prbstyle_v1/prbstyle_v1.safetensors", 0.8)],
              refs=["skizze.png"], batch=4, seed=1)
"""

DISTILLED = dict(unet="flux3-image-klein-bf16.safetensors", steps=16, cfg=1.0, sampler="euler", scheduler="simple")
BASE = dict(unet="flux3-image-klein-base-bf16.safetensors", steps=28, cfg=3.5, sampler="euler", scheduler="simple")


def graph(prompt, loras=(), refs=(), batch=4, seed=1, width=1024, height=1280, megapixels=1.5,
          model=DISTILLED, prefix="relay/klein", negative="",
          text_encoder="qwen3vl_4b_fp8_scaled.safetensors"):
    g, n = {}, [0]

    def add(cls, **inputs):
        n[0] += 1
        g[str(n[0])] = {"class_type": cls, "inputs": inputs}
        return [str(n[0]), 0]

    m = add("UNETLoader", unet_name=model["unet"], weight_dtype="default")
    for name, strength in loras:
        m = add("LoraLoaderModelOnly", model=m, lora_name=name, strength_model=strength)
    clip = add("CLIPLoader", clip_name=text_encoder, type="flux3", device="default")
    vae = add("VAELoader", vae_name="flux2-vae.safetensors")
    pos = add("CLIPTextEncode", clip=clip, text=prompt)
    if negative and model["cfg"] > 1:
        neg = add("CLIPTextEncode", clip=clip, text=negative)
    else:
        neg = add("ConditioningZeroOut", conditioning=pos)
    if refs:
        ref_in = {"positive": pos, "negative": neg, "vae": vae, "resolution": "resize",
                  "resolution.megapixels": megapixels}
        for i, r in enumerate(refs):
            ref_in[f"images.image{i}"] = add("LoadImage", image=r)
        rid = add("ReferenceMultiLatent", **ref_in)[0]
        pos, neg = [rid, 0], [rid, 1]
        lat = add("RepeatLatentBatch", samples=[rid, 2], amount=batch)
    else:
        lat = add("EmptyFlux2LatentImage", width=width, height=height, batch_size=batch)
    ks = add("KSampler", model=m, positive=pos, negative=neg, latent_image=lat, seed=seed,
             steps=model["steps"], cfg=model["cfg"], sampler_name=model["sampler"],
             scheduler=model["scheduler"], denoise=1.0)
    img = add("VAEDecode", samples=ks, vae=vae)
    add("SaveImage", images=img, filename_prefix=prefix)
    return g
