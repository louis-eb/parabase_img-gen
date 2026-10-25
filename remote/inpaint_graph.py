"""Re-paint one region of an existing image with FLUX.3 Klein (+ LoRAs), e.g. to let the model invent a joint where
two parts meet. The mask (white = re-paint) is scaled to the base image; outside the mask the base stays untouched.

    from inpaint_graph import inpaint
    g = inpaint("base.png", "maske_knoten1.png", "mtbflift, ... joined by a forged steel saddle ...",
                loras=[("mtbflift_v1/mtbflift_v1.safetensors", 0.95)], denoise=0.9)
"""


def inpaint(base, mask, prompt, loras=(), denoise=0.9, batch=2, seed=1, prefix="G_inpaint/knoten",
            unet="flux3-image-klein-bf16.safetensors", text_encoder="qwen3vl_4b_int8_convrot.safetensors",
            steps=16):
    g, n = {}, [0]

    def add(cls, **inputs):
        n[0] += 1
        g[str(n[0])] = {"class_type": cls, "inputs": inputs}
        return [str(n[0]), 0]

    m = add("UNETLoader", unet_name=unet, weight_dtype="default")
    for name, s in loras:
        m = add("LoraLoaderModelOnly", model=m, lora_name=name, strength_model=s)
    clip = add("CLIPLoader", clip_name=text_encoder, type="flux3", device="default")
    vae = add("VAELoader", vae_name="flux2-vae.safetensors")
    img = add("LoadImage", image=base)
    size = add("GetImageSize", image=img)
    mimg = add("LoadImage", image=mask)
    mimg = add("ImageScale", image=mimg, upscale_method="bilinear", width=[size[0], 0], height=[size[0], 1], crop="disabled")
    msk = add("ImageToMask", image=mimg, channel="red")
    pos = add("CLIPTextEncode", clip=clip, text=prompt)
    neg = add("ConditioningZeroOut", conditioning=pos)
    lat = add("VAEEncode", pixels=img, vae=vae)
    lat = add("SetLatentNoiseMask", samples=lat, mask=msk)
    lat = add("RepeatLatentBatch", samples=lat, amount=batch)
    ks = add("KSampler", model=m, positive=pos, negative=neg, latent_image=lat, seed=seed, steps=steps, cfg=1.0,
             sampler_name="euler", scheduler="simple", denoise=denoise)
    dec = add("VAEDecode", samples=ks, vae=vae)
    out = add("ImageCompositeMasked", destination=img, source=dec, x=0, y=0, resize_source=False, mask=msk)
    add("SaveImage", images=out, filename_prefix=prefix)
    return g
