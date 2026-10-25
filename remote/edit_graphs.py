"""Qwen-Image-Edit-2511 (input image -> edited image, geometry follows the input) and FLUX.2 [dev] with one reference image."""


def _b():
    g, n = {}, [0]

    def add(cls, **inputs):
        n[0] += 1
        g[str(n[0])] = {"class_type": cls, "inputs": inputs}
        return [str(n[0]), 0]
    return g, add


def qwen_edit(image, prompt, negative="", seed=1, steps=30, cfg=4.0, mp=1.2, prefix="Q_edit/qwen_edit",
              unet="qwen_image_edit_2511_fp8mixed.safetensors"):
    g, add = _b()
    m = add("UNETLoader", unet_name=unet, weight_dtype="default")
    clip = add("CLIPLoader", clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors", type="qwen_image", device="default")
    vae = add("VAELoader", vae_name="qwen_image_vae.safetensors")
    img = add("ImageScaleToTotalPixels", image=add("LoadImage", image=image), upscale_method="lanczos", megapixels=mp, resolution_steps=16)
    pos = add("TextEncodeQwenImageEditPlus", clip=clip, prompt=prompt, vae=vae, image1=img)
    neg = add("TextEncodeQwenImageEditPlus", clip=clip, prompt=negative, vae=vae, image1=img)
    ms = add("ModelSamplingAuraFlow", model=m, shift=3.0)
    ms = add("CFGNorm", model=ms, strength=1.0)
    lat = add("VAEEncode", pixels=img, vae=vae)
    ks = add("KSampler", model=ms, positive=pos, negative=neg, latent_image=lat, seed=seed, steps=steps, cfg=cfg,
             sampler_name="euler", scheduler="simple", denoise=1.0)
    add("SaveImage", images=add("VAEDecode", samples=ks, vae=vae), filename_prefix=prefix)
    return g


def flux2_dev(ref, prompt, extra_refs=(), loras=(), init=None, denoise=1.0, sampler_name="euler", seed=1, steps=28, guidance=4.0, mp=1.4, prefix="F_flux2/flux2",
              unet="flux2_dev_fp8mixed.safetensors", te="mistral_3_small_flux2_fp4_mixed.safetensors"):
    g, add = _b()
    m = add("UNETLoader", unet_name=unet, weight_dtype="default")
    for name, st in loras:
        m = add("LoraLoaderModelOnly", model=m, lora_name=name, strength_model=st)
    clip = add("CLIPLoader", clip_name=te, type="flux2", device="default")
    vae = add("VAELoader", vae_name="flux2-vae.safetensors")
    img = add("ImageScaleToTotalPixels", image=add("LoadImage", image=ref), upscale_method="lanczos", megapixels=mp, resolution_steps=16)
    size = add("GetImageSize", image=img)
    pos = add("CLIPTextEncode", clip=clip, text=prompt)
    pos = add("ReferenceLatent", conditioning=pos, latent=add("VAEEncode", pixels=img, vae=vae))
    for r in extra_refs:
        ri = add("ImageScaleToTotalPixels", image=add("LoadImage", image=r), upscale_method="lanczos", megapixels=mp, resolution_steps=16)
        pos = add("ReferenceLatent", conditioning=pos, latent=add("VAEEncode", pixels=ri, vae=vae))
    pos = add("FluxGuidance", conditioning=pos, guidance=guidance)
    guider = add("BasicGuider", model=m, conditioning=pos)
    sampler = add("KSamplerSelect", sampler_name=sampler_name)
    sig = add("Flux2Scheduler", steps=steps, width=[size[0], 0], height=[size[0], 1])
    noise = add("RandomNoise", noise_seed=seed)
    lat = add("EmptyFlux2LatentImage", width=[size[0], 0], height=[size[0], 1], batch_size=1)
    if init:  # image-to-image: start from an existing image, only the last part of the schedule
        ii = add("ImageScale", image=add("LoadImage", image=init), upscale_method="lanczos", width=[size[0], 0], height=[size[0], 1], crop="disabled")
        lat = add("VAEEncode", pixels=ii, vae=vae)
        sig = [add("SplitSigmasDenoise", sigmas=sig, denoise=denoise)[0], 1]
    out = add("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sig, latent_image=lat)
    add("SaveImage", images=add("VAEDecode", samples=out, vae=vae), filename_prefix=prefix)
    return g
