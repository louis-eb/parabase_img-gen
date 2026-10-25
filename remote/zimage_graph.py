"""Z-Image Turbo + Fun ControlNet Union 2.1 as API graph: the geometry comes exactly from a
line render (or depth/clay render) of the 3D scene. Use the result as input for Klein + LoRA
(klein_graph(..., init=..., denoise=0.4)) to bring in the salvaged lift parts.

    from zimage_graph import zimage_cn
    g = zimage_cn("linien.png", "documentary photograph of ...", strength=0.8, invert=True)
"""


def zimage_cn(control_image, prompt, strength=0.8, end=0.8, batch=4, seed=7, mp=1.5, invert=True,
              prefix="A_3d/zimage"):
    g, n = {}, [0]

    def add(cls, **inputs):
        n[0] += 1
        g[str(n[0])] = {"class_type": cls, "inputs": inputs}
        return [str(n[0]), 0]

    m = add("UNETLoader", unet_name="z_image_turbo_bf16.safetensors", weight_dtype="default")
    clip = add("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="lumina2", device="default")
    vae = add("VAELoader", vae_name="ae.safetensors")
    patch = add("ModelPatchLoader", name="Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors")
    img = add("LoadImage", image=control_image)
    img = add("ImageScaleToTotalPixels", image=img, upscale_method="lanczos", megapixels=mp, resolution_steps=16)
    ctrl = img
    if invert:  # black lines on white -> white lines on black, as the scribble/lineart control expects
        ctrl = add("ImageInvert", image=img)
    pos = add("CLIPTextEncode", clip=clip, text=prompt)
    neg = add("ConditioningZeroOut", conditioning=pos)
    cn = add("QwenImageDiffsynthControlnet", model=m, model_patch=patch, vae=vae, image=ctrl, strength=strength,
             start_percent=0.0, end_percent=end)
    ms = add("ModelSamplingAuraFlow", model=cn, shift=3)
    size = add("GetImageSize", image=img)
    lat = add("EmptySD3LatentImage", width=[size[0], 0], height=[size[0], 1], batch_size=batch)
    ks = add("KSampler", model=ms, positive=pos, negative=neg, latent_image=lat, seed=seed, steps=8, cfg=1,
             sampler_name="res_multistep", scheduler="simple", denoise=1.0)
    dec = add("VAEDecode", samples=ks, vae=vae)
    add("SaveImage", images=dec, filename_prefix=prefix)
    return g
