"""Minimal builder for ComfyUI UI-format (LiteGraph v0.4) workflows."""
import json


class WF:
    def __init__(self):
        self.nodes, self.links, self.groups = [], [], []
        self.nid, self.lid, self.gid = 0, 0, 0

    def node(self, type_, pos, widgets=None, inputs=(), outputs=(), size=(320, 120),
             title=None, color=None, bgcolor=None, mode=0):
        """inputs: list of (name, type) or (name, type, widget_name). outputs: list of (name, type)."""
        self.nid += 1
        n = {
            "id": self.nid, "type": type_, "pos": list(pos), "size": list(size),
            "flags": {}, "order": self.nid, "mode": mode,
            "inputs": [], "outputs": [],
            "properties": {"Node name for S&R": type_, "cnr_id": "comfy-core"},
        }
        if type_ in ("MarkdownNote", "Note"):
            n["properties"] = {}
        for inp in inputs:
            d = {"localized_name": inp[0], "name": inp[0], "type": inp[1], "link": None}
            if len(inp) > 2 and inp[2]:
                d["widget"] = {"name": inp[2]}
            n["inputs"].append(d)
        for i, (name, typ) in enumerate(outputs):
            n["outputs"].append({"localized_name": name, "name": name, "type": typ,
                                 "slot_index": i, "links": []})
        if widgets is not None:
            n["widgets_values"] = widgets
        if title:
            n["title"] = title
        if color:
            n["color"] = color
        if bgcolor:
            n["bgcolor"] = bgcolor
        self.nodes.append(n)
        return n

    def link(self, src, src_slot, dst, dst_input):
        """src_slot: int or output name; dst_input: input name."""
        if isinstance(src_slot, str):
            src_slot = [o["name"] for o in src["outputs"]].index(src_slot)
        names = [i["name"] for i in dst["inputs"]]
        if dst_input not in names:
            raise KeyError(f"{dst['type']} has no input {dst_input}: {names}")
        dslot = names.index(dst_input)
        typ = src["outputs"][src_slot]["type"]
        self.lid += 1
        self.links.append([self.lid, src["id"], src_slot, dst["id"], dslot, typ])
        src["outputs"][src_slot]["links"].append(self.lid)
        if dst["inputs"][dslot]["link"] is not None:
            raise ValueError(f"input {dst_input} on {dst['type']} already linked")
        dst["inputs"][dslot]["link"] = self.lid
        return self.lid

    def note(self, pos, text, size=(420, 300), color="#432", bgcolor="#653"):
        return self.node("MarkdownNote", pos, widgets=[text], size=size, color=color, bgcolor=bgcolor)

    def group(self, title, x, y, w, h, color="#3f789e"):
        self.gid += 1
        self.groups.append({"id": self.gid, "title": title, "bounding": [x, y, w, h],
                            "color": color, "font_size": 24, "flags": {}})

    def to_json(self):
        for n in self.nodes:
            for o in n["outputs"]:
                if not o["links"]:
                    o["links"] = None
        return {
            "id": "00000000-0000-4000-8000-00000000000" + str(len(self.nodes) % 10),
            "revision": 0,
            "last_node_id": self.nid, "last_link_id": self.lid,
            "nodes": self.nodes, "links": self.links, "groups": self.groups,
            "config": {}, "extra": {"ds": {"scale": 0.6, "offset": [0, 0]}},
            "version": 0.4,
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=1)


# ---------- typed node helpers (inputs/outputs mirror current ComfyUI core schemas) ----------

def unet(wf, pos, name):
    return wf.node("UNETLoader", pos, [name, "default"], outputs=[("MODEL", "MODEL")], size=(360, 82))

def clip(wf, pos, name, typ):
    return wf.node("CLIPLoader", pos, [name, typ, "default"], outputs=[("CLIP", "CLIP")], size=(360, 106))

def vae(wf, pos, name):
    return wf.node("VAELoader", pos, [name], outputs=[("VAE", "VAE")], size=(360, 58))

def load_image(wf, pos, filename, title=None):
    return wf.node("LoadImage", pos, [filename, "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                   size=(320, 380), title=title)

def text_block(wf, pos, text, title, size=(460, 170)):
    return wf.node("PrimitiveStringMultiline", pos, [text], outputs=[("STRING", "STRING")], size=size, title=title,
                   color="#232", bgcolor="#353")

def concat(wf, pos, delimiter=" "):
    return wf.node("StringConcatenate", pos, ["", "", delimiter],
                   inputs=[("string_a", "STRING", "string_a"), ("string_b", "STRING", "string_b")],
                   outputs=[("STRING", "STRING")], size=(260, 120))

def clip_encode(wf, pos, title=None):
    return wf.node("CLIPTextEncode", pos, [""], inputs=[("clip", "CLIP"), ("text", "STRING", "text")],
                   outputs=[("CONDITIONING", "CONDITIONING")], size=(300, 90), title=title)

def zero_out(wf, pos):
    return wf.node("ConditioningZeroOut", pos, [], inputs=[("conditioning", "CONDITIONING")],
                   outputs=[("CONDITIONING", "CONDITIONING")], size=(240, 30))

def ksampler(wf, pos, seed, steps, cfg, sampler, scheduler, denoise, control="randomize"):
    return wf.node("KSampler", pos, [seed, control, steps, cfg, sampler, scheduler, denoise],
                   inputs=[("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                           ("latent_image", "LATENT")],
                   outputs=[("LATENT", "LATENT")], size=(320, 262))

def vae_decode(wf, pos):
    return wf.node("VAEDecode", pos, [], inputs=[("samples", "LATENT"), ("vae", "VAE")],
                   outputs=[("IMAGE", "IMAGE")], size=(200, 46))

def vae_encode(wf, pos):
    return wf.node("VAEEncode", pos, [], inputs=[("pixels", "IMAGE"), ("vae", "VAE")],
                   outputs=[("LATENT", "LATENT")], size=(200, 46))

def save_image(wf, pos, prefix, size=(520, 560)):
    return wf.node("SaveImage", pos, [prefix], inputs=[("images", "IMAGE")], size=size)

def preview(wf, pos, size=(320, 340), title=None):
    return wf.node("PreviewImage", pos, [], inputs=[("images", "IMAGE")], size=size, title=title)

def scale_total(wf, pos, mp, steps=16, method="lanczos"):
    return wf.node("ImageScaleToTotalPixels", pos, [method, mp, steps], inputs=[("image", "IMAGE")],
                   outputs=[("IMAGE", "IMAGE")], size=(300, 106))
