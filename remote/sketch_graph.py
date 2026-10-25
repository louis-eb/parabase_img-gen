"""Sketch -> photo in one ComfyUI graph: Z-Image Turbo + Fun ControlNet takes the geometry from a line
sketch, then FLUX.3 Klein with our LoRAs re-draws the result image-to-image (A -> F chained).

    from sketch_graph import sketch_chain
    g = sketch_chain("skizze.png", zimage_prompt, klein_prompt, loras=[("mtbflift_v1/mtbflift_v1.safetensors", 0.7)])
"""


def sketch_chain(control_image, zimage_prompt, klein_prompt, loras=(), strength=0.8, end=0.8, denoise=0.55,
                 batch=2, seed=7, mp=1.5, invert=True, prefix="G_skizze/chain",
                 klein_unet="flux3-image-klein-bf16.safetensors",
                 text_encoder="qwen3vl_4b_int8_convrot.safetensors", save_stage1=True):
    g, n = {}, [0]

    def add(cls, **inputs):
        n[0] += 1
        g[str(n[0])] = {"class_type": cls, "inputs": inputs}
        return [str(n[0]), 0]

    # stage 1: geometry from the sketch (Z-Image + ControlNet)
    zm = add("UNETLoader", unet_name="z_image_turbo_bf16.safetensors", weight_dtype="default")
    zclip = add("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="lumina2", device="default")
    zvae = add("VAELoader", vae_name="ae.safetensors")
    patch = add("ModelPatchLoader", name="Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors")
    img = add("LoadImage", image=control_image)
    img = add("ImageScaleToTotalPixels", image=img, upscale_method="lanczos", megapixels=mp, resolution_steps=16)
    ctrl = add("ImageInvert", image=img) if invert else img
    zpos = add("CLIPTextEncode", clip=zclip, text=zimage_prompt)
    zneg = add("ConditioningZeroOut", conditioning=zpos)
    cn = add("QwenImageDiffsynthControlnet", model=zm, model_patch=patch, vae=zvae, image=ctrl, strength=strength,
             start_percent=0.0, end_percent=end)
    ms = add("ModelSamplingAuraFlow", model=cn, shift=3)
    size = add("GetImageSize", image=img)
    lat = add("EmptySD3LatentImage", width=[size[0], 0], height=[size[0], 1], batch_size=batch)
    ks = add("KSampler", model=ms, positive=zpos, negative=zneg, latent_image=lat, seed=seed, steps=8, cfg=1,
             sampler_name="res_multistep", scheduler="simple", denoise=1.0)
    stage1 = add("VAEDecode", samples=ks, vae=zvae)
    if save_stage1:
        add("SaveImage", images=stage1, filename_prefix=prefix + "_stufe1")

    # stage 2: Klein + LoRAs re-draws it (materials, salvaged parts, style)
    km = add("UNETLoader", unet_name=klein_unet, weight_dtype="default")
    for name, s in loras:
        km = add("LoraLoaderModelOnly", model=km, lora_name=name, strength_model=s)
    kclip = add("CLIPLoader", clip_name=text_encoder, type="flux3", device="default")
    kvae = add("VAELoader", vae_name="flux2-vae.safetensors")
    kpos = add("CLIPTextEncode", clip=kclip, text=klein_prompt)
    kneg = add("ConditioningZeroOut", conditioning=kpos)
    klat = add("VAEEncode", pixels=stage1, vae=kvae)
    ks2 = add("KSampler", model=km, positive=kpos, negative=kneg, latent_image=klat, seed=seed, steps=16, cfg=1.0,
              sampler_name="euler", scheduler="simple", denoise=denoise)
    add("SaveImage", images=add("VAEDecode", samples=ks2, vae=kvae), filename_prefix=prefix)
    return g
