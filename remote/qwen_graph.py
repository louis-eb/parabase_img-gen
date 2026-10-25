"""Qwen-Image-2512 (20B, fp8) with the DiffSynth canny ControlNet patch: geometry from a line drawing.

    from qwen_graph import qwen_cn
    g = qwen_cn("linien.png", "architectural photograph of ...", strength=0.9)
"""


def qwen_cn(control_image, prompt, negative="", strength=0.9, batch=1, seed=1, mp=1.4, steps=30, cfg=3.0,
            invert=True, prefix="Q_qwen/qwen", unet="qwen_image_2512_fp8_e4m3fn.safetensors", control=True, loras=()):
    g, n = {}, [0]

    def add(cls, **inputs):
        n[0] += 1
        g[str(n[0])] = {"class_type": cls, "inputs": inputs}
        return [str(n[0]), 0]

    m = add("UNETLoader", unet_name=unet, weight_dtype="default")
    for name, s in loras:
        m = add("LoraLoaderModelOnly", model=m, lora_name=name, strength_model=s)
    clip = add("CLIPLoader", clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors", type="qwen_image", device="default")
    vae = add("VAELoader", vae_name="qwen_image_vae.safetensors")
    img = add("LoadImage", image=control_image)
    img = add("ImageScaleToTotalPixels", image=img, upscale_method="lanczos", megapixels=mp, resolution_steps=16)
    if control:
        ctrl = add("ImageInvert", image=img) if invert else img
        patch = add("ModelPatchLoader", name="qwen_image_canny_diffsynth_controlnet.safetensors")
        m = add("QwenImageDiffsynthControlnet", model=m, model_patch=patch, vae=vae, image=ctrl, strength=strength)
    ms = add("ModelSamplingAuraFlow", model=m, shift=3.1)
    pos = add("CLIPTextEncode", clip=clip, text=prompt)
    neg = add("CLIPTextEncode", clip=clip, text=negative)
    size = add("GetImageSize", image=img)
    lat = add("EmptySD3LatentImage", width=[size[0], 0], height=[size[0], 1], batch_size=batch)
    ks = add("KSampler", model=ms, positive=pos, negative=neg, latent_image=lat, seed=seed, steps=steps, cfg=cfg,
             sampler_name="euler", scheduler="simple", denoise=1.0)
    add("SaveImage", images=add("VAEDecode", samples=ks, vae=vae), filename_prefix=prefix)
    return g
