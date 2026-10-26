"""ComfyUI API graph for Workflow C (Z-Image detail pass + SeedVR2 7B), for 'comfy' jobs.

    from upscale_graph import upscale
    g = upscale("up_green.png", "documentary photograph of ...", denoise=0.3, mp=4.0, mult=2.0)

With SETUP=flux2 (no Z-Image on the instance) use denoise=0: SeedVR2 only.
"""


def upscale(image, prompt, denoise=0.3, mp=4.0, mult=2.0, seed=20261019, prefix="C_upscale/final"):
    g, n = {}, [0]

    def add(cls, **inputs):
        n[0] += 1
        g[str(n[0])] = {"class_type": cls, "inputs": inputs}
        return [str(n[0]), 0]

    m = add("UNETLoader", unet_name="z_image_turbo_bf16.safetensors", weight_dtype="default")
    clip = add("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="lumina2", device="default")
    vae = add("VAELoader", vae_name="ae.safetensors")
    upm = add("UpscaleModelLoader", model_name="RealESRGAN_x4plus.safetensors")
    sm = add("UNETLoader", unet_name="seedvr2_7b_int8_convrot.safetensors", weight_dtype="default")
    svae = add("VAELoader", vae_name="seedvr2_ema_vae_fp16.safetensors")
    img = add("LoadImage", image=image)
    pos = add("CLIPTextEncode", clip=clip, text=prompt)
    neg = add("ConditioningZeroOut", conditioning=pos)
    if denoise > 0:
        ups = add("ImageUpscaleWithModel", upscale_model=upm, image=img)
        sc = add("ImageScaleToTotalPixels", image=ups, upscale_method="lanczos", megapixels=mp, resolution_steps=16)
        enc = add("VAEEncode", pixels=sc, vae=vae)
        ms = add("ModelSamplingAuraFlow", model=m, shift=3)
        ks = add("KSampler", model=ms, positive=pos, negative=neg, latent_image=enc, seed=seed, steps=5, cfg=1,
                 sampler_name="dpmpp_2m_sde", scheduler="beta", denoise=denoise)
        stage1 = add("VAEDecode", samples=ks, vae=vae)
    else:
        stage1 = add("ImageScaleToTotalPixels", image=img, upscale_method="lanczos", megapixels=mp, resolution_steps=16)
    rs = add("ResizeImageMaskNode", input=stage1, resize_type="scale by multiplier",
             **{"resize_type.multiplier": mult}, scale_method="lanczos")
    pre = add("SeedVR2Preprocess", resized_images=rs)
    et = add("VAEEncodeTiled", pixels=pre, vae=svae, tile_size=512, overlap=128, temporal_size=4096, temporal_overlap=8)
    cond = add("SeedVR2Conditioning", model=sm, vae_conditioning=et)
    ks2 = add("KSampler", model=sm, positive=cond, negative=[cond[0], 1], latent_image=et, seed=seed, steps=1, cfg=1,
              sampler_name="euler", scheduler="simple", denoise=1.0)
    dt = add("VAEDecodeTiled", samples=ks2, vae=svae, tile_size=512, overlap=128, temporal_size=4096, temporal_overlap=8)
    post = add("SeedVR2PostProcessing", images=dt, original_resized_images=rs, color_correction_method="lab")
    add("SaveImage", images=post, filename_prefix=prefix)
    return g
