#!/bin/bash
# =============================================================================
#  Architektur-Stimmungen · ComfyUI Provisioning für vast.ai
#  Template: offizielles vast.ai "ComfyUI" Template  (Image vastai/comfy)
#  Umgebungsvariable:  PROVISIONING_SCRIPT = <raw-URL dieser Datei>
#                      HF_TOKEN            = <dein Hugging-Face-Token>  (für FLUX.3 Klein nötig)
#
#  Was das Skript macht:
#   1. ComfyUI auf den neuesten Stand bringen (FLUX.3 Klein braucht ComfyUI >= Okt. 2026)
#   2. Alle Modelle parallel herunterladen (~50 GB, überspringt vorhandene Dateien)
#   3. Die drei Workflows A/B/C in den Workflow-Browser legen
#   4. Beispiel-Skizze und Beispiel-Materialien in den input-Ordner legen
#  Keine Custom Nodes nötig – alles läuft mit ComfyUI-Core-Nodes.
# =============================================================================
set -uo pipefail

# ComfyUI-venv des vast.ai-Images aktivieren
# shellcheck disable=SC1091
source /venv/main/bin/activate

WORKSPACE_DIR="${WORKSPACE:-/workspace}"
COMFYUI_DIR="${WORKSPACE_DIR}/ComfyUI"
MODELS_DIR="${COMFYUI_DIR}/models"
WORKFLOW_DIR="${COMFYUI_DIR}/user/default/workflows"
INPUT_DIR="${COMFYUI_DIR}/input"
LOG="${WORKSPACE_DIR}/provisioning_arch.log"
MAX_PARALLEL=3

# "repo|pfad/im/repo|zielordner"
MODELS=(
  # --- Z-Image Turbo (Workflow A + Detail-Pass in C) — Apache 2.0
  "Comfy-Org/z_image_turbo|split_files/diffusion_models/z_image_turbo_bf16.safetensors|diffusion_models"
  "Comfy-Org/z_image_turbo|split_files/text_encoders/qwen_3_4b.safetensors|text_encoders"
  "Comfy-Org/z_image_turbo|split_files/vae/ae.safetensors|vae"
  # --- Fun ControlNet Union 2.1 (2602, 8 Steps): Canny/HED/Depth/MLSD/Scribble/Gray
  "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1|Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors|model_patches"
  # --- FLUX.3 Image Klein (Workflow B) — gated: Lizenz auf HF akzeptieren + HF_TOKEN setzen
  "black-forest-labs/flux-3-image-klein-comfy|diffusion_models/flux3-image-klein-bf16.safetensors|diffusion_models"
  "black-forest-labs/flux-3-image-klein-comfy|text_encoders/qwen3vl_4b_int8_convrot.safetensors|text_encoders"
  "black-forest-labs/flux-3-image-klein-comfy|vae/flux2-vae.safetensors|vae"
  # --- Upscale (Workflow C)
  "Comfy-Org/Real-ESRGAN_repackaged|RealESRGAN_x4plus.safetensors|upscale_models"
  "Comfy-Org/SeedVR2|diffusion_models/seedvr2_7b_int8_convrot.safetensors|diffusion_models"
  "Comfy-Org/SeedVR2|vae/seedvr2_ema_vae_fp16.safetensors|vae"
)

log() { echo "[arch-provisioning] $*" | tee -a "$LOG"; }

# -----------------------------------------------------------------------------
update_comfyui() {
  log "ComfyUI aktualisieren …"
  cd "$COMFYUI_DIR" || { log "FEHLER: $COMFYUI_DIR nicht gefunden"; return 1; }
  git fetch --tags origin >>"$LOG" 2>&1
  if ! git symbolic-ref -q HEAD >/dev/null; then
    git checkout -q master >>"$LOG" 2>&1 || git checkout -q -B master origin/master >>"$LOG" 2>&1
  fi
  git pull --ff-only >>"$LOG" 2>&1 || git reset --hard origin/master >>"$LOG" 2>&1
  if command -v uv >/dev/null 2>&1; then
    uv pip install --no-cache-dir -r requirements.txt >>"$LOG" 2>&1
  else
    pip install --no-cache-dir -r requirements.txt >>"$LOG" 2>&1
  fi
  log "ComfyUI-Version: $(git describe --tags --always 2>/dev/null)"
}

# -----------------------------------------------------------------------------
download_one() {
  local repo="$1" path="$2" sub="$3"
  local dest_dir="${MODELS_DIR}/${sub}" name; name="$(basename "$path")"
  local dest="${dest_dir}/${name}"
  mkdir -p "$dest_dir"
  if [[ -s "$dest" ]]; then log "vorhanden: ${sub}/${name}"; return 0; fi
  local tmp; tmp="$(mktemp -d "${WORKSPACE_DIR}/.dl_XXXX")"
  local try
  for try in 1 2 3 4; do
    if hf download "$repo" "$path" --local-dir "$tmp" >>"$LOG" 2>&1; then
      mv "${tmp}/${path}" "$dest" && rm -rf "$tmp"
      log "OK: ${sub}/${name}"; return 0
    fi
    log "Versuch ${try} fehlgeschlagen: ${repo}/${path} – neuer Versuch …"; sleep $((try * 5))
  done
  # Fallback: manche Repos benennen Dateien mit "_" statt "-" – unter dem erwarteten Namen speichern
  local alt; alt="$(dirname "$path")/$(basename "$path" | tr '-' '_')"; alt="${alt#./}"
  if [[ "$alt" != "$path" ]] && hf download "$repo" "$alt" --local-dir "$tmp" >>"$LOG" 2>&1; then
    mv "${tmp}/${alt}" "$dest" && rm -rf "$tmp"
    log "OK (alternativer Dateiname ${alt}): ${sub}/${name}"; return 0
  fi
  rm -rf "$tmp"; log "FEHLER: ${repo}/${path} konnte nicht geladen werden"; return 1
}
export -f download_one log
export MODELS_DIR WORKSPACE_DIR LOG

download_models() {
  log "huggingface_hub (hf CLI) aktualisieren …"
  pip install --no-cache-dir -U "huggingface_hub[hf_xet]" >>"$LOG" 2>&1
  if [[ -z "${HF_TOKEN:-}" ]]; then
    log "WARNUNG: HF_TOKEN ist nicht gesetzt – FLUX.3 Klein (Workflow B) wird fehlen."
  else
    export HF_TOKEN
  fi
  export HF_HUB_ENABLE_HF_TRANSFER=0
  printf '%s\n' "${MODELS[@]}" | xargs -P "$MAX_PARALLEL" -I{} bash -c 'IFS="|" read -r r p s <<< "{}"; download_one "$r" "$p" "$s"'

  # Fallback: falls die bf16-Version von Klein nicht verfügbar ist, die int8-Variante nehmen
  if [[ ! -s "${MODELS_DIR}/diffusion_models/flux3-image-klein-bf16.safetensors" && -n "${HF_TOKEN:-}" ]]; then
    log "bf16 Klein fehlt – lade int8-Variante als Ersatz"
    if download_one "black-forest-labs/flux-3-image-klein-comfy" "diffusion_models/flux3-image-klein-int8-convrot.safetensors" "diffusion_models"; then
      sed -i 's/flux3-image-klein-bf16.safetensors/flux3-image-klein-int8-convrot.safetensors/g' \
        "${WORKFLOW_DIR}/B_Skizze_plus_Materialbilder.json" 2>/dev/null
    fi
  fi
}

# -----------------------------------------------------------------------------
write_workflows() {
  mkdir -p "$WORKFLOW_DIR"
  log "Workflows schreiben nach ${WORKFLOW_DIR}"
  cat > "${WORKFLOW_DIR}/A_Skizze_zu_Stimmung.json" <<'JSONEOF'
{
 "id": "00000000-0000-4000-8000-000000000005",
 "revision": 0,
 "last_node_id": 25,
 "last_link_id": 28,
 "nodes": [
  {
   "id": 1,
   "type": "MarkdownNote",
   "pos": [
    -1000,
    -40
   ],
   "size": [
    560,
    640
   ],
   "flags": {},
   "order": 1,
   "mode": 0,
   "inputs": [],
   "outputs": [],
   "properties": {},
   "widgets_values": [
    "## A · Skizze → Stimmung\n**Z-Image Turbo + Fun ControlNet Union 2.1 (Scribble)**\n\nSo arbeitest du damit:\n1. Skizze im Node **Skizze** hochladen (Bleistift, Fineliner, Screenshot – egal). Hochformat/Querformat wird übernommen.\n2. Die drei grünen Text-Blöcke anpassen: **Motiv**, **Material**, **Stimmung & Kamera**. Sie werden automatisch zu einem Prompt zusammengesetzt.\n3. **Run** – du bekommst 4 Varianten (Batch-Grösse im Node *Bildgrösse & Anzahl*).\n4. Das beste Bild per Rechtsklick → *Save Image* sichern und in Workflow **C · Upscale** laden.\n\n**Wichtigste Regler**\n- **ControlNet-Stärke** (Node *Skizze → Geometrie*): 0.6 = lockere Interpretation, 0.9 = sehr nah an der Skizze. Start: 0.75.\n- **end_percent** (unter *advanced*): 0.8–0.9 lässt dem Modell in den letzten Schritten Freiheit für Materialdetails.\n- **Skizze direkt / Canny** (Schalter): *false* = Skizze wird invertiert und als Scribble benutzt (für Handskizzen). *true* = Canny-Kanten (für saubere Strichzeichnungen oder Screenshots aus dem 3D-Modell).\n- **Seed** auf *fixed* stellen, wenn du an einem Bild nur den Prompt verfeinern willst.\n\nSteps 8 · CFG 1 · res_multistep/simple · Shift 3 sind auf das 8-Step-Modell abgestimmt – nicht verändern."
   ],
   "color": "#223",
   "bgcolor": "#335"
  },
  {
   "id": 2,
   "type": "UNETLoader",
   "pos": [
    -980,
    690
   ],
   "size": [
    360,
    82
   ],
   "flags": {},
   "order": 2,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "MODEL",
     "name": "MODEL",
     "type": "MODEL",
     "slot_index": 0,
     "links": [
      14
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "UNETLoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "z_image_turbo_bf16.safetensors",
    "default"
   ]
  },
  {
   "id": 3,
   "type": "CLIPLoader",
   "pos": [
    -980,
    800
   ],
   "size": [
    360,
    106
   ],
   "flags": {},
   "order": 3,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "CLIP",
     "name": "CLIP",
     "type": "CLIP",
     "slot_index": 0,
     "links": [
      11
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "CLIPLoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "qwen_3_4b.safetensors",
    "lumina2",
    "default"
   ]
  },
  {
   "id": 4,
   "type": "VAELoader",
   "pos": [
    -980,
    930
   ],
   "size": [
    360,
    58
   ],
   "flags": {},
   "order": 4,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "VAE",
     "name": "VAE",
     "type": "VAE",
     "slot_index": 0,
     "links": [
      16,
      27
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "VAELoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "ae.safetensors"
   ]
  },
  {
   "id": 5,
   "type": "ModelPatchLoader",
   "pos": [
    -980,
    1010
   ],
   "size": [
    400,
    58
   ],
   "flags": {},
   "order": 5,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "MODEL_PATCH",
     "name": "MODEL_PATCH",
     "type": "MODEL_PATCH",
     "slot_index": 0,
     "links": [
      15
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ModelPatchLoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"
   ]
  },
  {
   "id": 6,
   "type": "LoadImage",
   "pos": [
    -380,
    10
   ],
   "size": [
    320,
    380
   ],
   "flags": {},
   "order": 6,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      1
     ]
    },
    {
     "localized_name": "MASK",
     "name": "MASK",
     "type": "MASK",
     "slot_index": 1,
     "links": null
    }
   ],
   "properties": {
    "Node name for S&R": "LoadImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "skizze_beispiel.png",
    "image"
   ],
   "title": "Skizze"
  },
  {
   "id": 7,
   "type": "ImageScaleToTotalPixels",
   "pos": [
    -40,
    10
   ],
   "size": [
    300,
    106
   ],
   "flags": {},
   "order": 7,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "image",
     "name": "image",
     "type": "IMAGE",
     "link": 1
    }
   ],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      2,
      3,
      19
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ImageScaleToTotalPixels",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "lanczos",
    1.5,
    16
   ],
   "title": "Skizze auf ~1.5 MP skalieren"
  },
  {
   "id": 8,
   "type": "ImageInvert",
   "pos": [
    -40,
    150
   ],
   "size": [
    300,
    30
   ],
   "flags": {},
   "order": 8,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "image",
     "name": "image",
     "type": "IMAGE",
     "link": 2
    }
   ],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      4
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ImageInvert",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [],
   "title": "Invertieren (Scribble)"
  },
  {
   "id": 9,
   "type": "Canny",
   "pos": [
    -40,
    220
   ],
   "size": [
    300,
    82
   ],
   "flags": {},
   "order": 9,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "image",
     "name": "image",
     "type": "IMAGE",
     "link": 3
    }
   ],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      5
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "Canny",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    0.15,
    0.35
   ],
   "title": "Canny (für saubere Linien)"
  },
  {
   "id": 10,
   "type": "ComfySwitchNode",
   "pos": [
    -40,
    340
   ],
   "size": [
    300,
    82
   ],
   "flags": {},
   "order": 10,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "on_false",
     "name": "on_false",
     "type": "IMAGE",
     "link": 4
    },
    {
     "localized_name": "on_true",
     "name": "on_true",
     "type": "IMAGE",
     "link": 5
    }
   ],
   "outputs": [
    {
     "localized_name": "output",
     "name": "output",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      6,
      17
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ComfySwitchNode",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    false
   ],
   "title": "Skizze direkt (false) / Canny (true)"
  },
  {
   "id": 11,
   "type": "PreviewImage",
   "pos": [
    300,
    10
   ],
   "size": [
    340,
    400
   ],
   "flags": {},
   "order": 11,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "images",
     "name": "images",
     "type": "IMAGE",
     "link": 6
    }
   ],
   "outputs": [],
   "properties": {
    "Node name for S&R": "PreviewImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [],
   "title": "Kontrollbild"
  },
  {
   "id": 12,
   "type": "PrimitiveStringMultiline",
   "pos": [
    -380,
    790
   ],
   "size": [
    460,
    170
   ],
   "flags": {},
   "order": 12,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "STRING",
     "name": "STRING",
     "type": "STRING",
     "slot_index": 0,
     "links": [
      7
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "PrimitiveStringMultiline",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "Close-up architectural photograph of a building fragment, exactly as drawn in the sketch: a deep window reveal cut into a thick wall, a slender lintel above, the edge of the opening seen slightly from below. Only a part of the building is visible, cropped tightly, no sky line, no full facade."
   ],
   "title": "Motiv / Fragment",
   "color": "#232",
   "bgcolor": "#353"
  },
  {
   "id": 13,
   "type": "PrimitiveStringMultiline",
   "pos": [
    -380,
    980
   ],
   "size": [
    460,
    170
   ],
   "flags": {},
   "order": 13,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "STRING",
     "name": "STRING",
     "type": "STRING",
     "slot_index": 0,
     "links": [
      8
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "PrimitiveStringMultiline",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "The wall is board-marked in-situ concrete with a fine timber grain imprint, small air pores, subtle efflorescence and darker weathering streaks below the lintel. The lintel is dark patinated steel with a faint rust bloom. The window is a deep frameless glass pane reflecting a soft grey sky."
   ],
   "title": "Material",
   "color": "#232",
   "bgcolor": "#353"
  },
  {
   "id": 14,
   "type": "PrimitiveStringMultiline",
   "pos": [
    -380,
    1170
   ],
   "size": [
    460,
    170
   ],
   "flags": {},
   "order": 14,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "STRING",
     "name": "STRING",
     "type": "STRING",
     "slot_index": 0,
     "links": [
      10
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "PrimitiveStringMultiline",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "Shot on a medium format camera, 80mm lens, f/5.6, eye-level tripod shot. Soft overcast late-afternoon light grazing across the surface, revealing every texture, deep soft shadows inside the reveal. Muted, desaturated palette of warm greys and dark umber, quiet, contemplative, almost still-life atmosphere, light haze, subtle film grain. Precise physical material detail, photorealistic architectural photography."
   ],
   "title": "Stimmung & Kamera",
   "color": "#232",
   "bgcolor": "#353"
  },
  {
   "id": 15,
   "type": "StringConcatenate",
   "pos": [
    110,
    860
   ],
   "size": [
    260,
    120
   ],
   "flags": {},
   "order": 15,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "string_a",
     "name": "string_a",
     "type": "STRING",
     "link": 7,
     "widget": {
      "name": "string_a"
     }
    },
    {
     "localized_name": "string_b",
     "name": "string_b",
     "type": "STRING",
     "link": 8,
     "widget": {
      "name": "string_b"
     }
    }
   ],
   "outputs": [
    {
     "localized_name": "STRING",
     "name": "STRING",
     "type": "STRING",
     "slot_index": 0,
     "links": [
      9
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "StringConcatenate",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "",
    "",
    " "
   ]
  },
  {
   "id": 16,
   "type": "StringConcatenate",
   "pos": [
    110,
    1060
   ],
   "size": [
    260,
    120
   ],
   "flags": {},
   "order": 16,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "string_a",
     "name": "string_a",
     "type": "STRING",
     "link": 9,
     "widget": {
      "name": "string_a"
     }
    },
    {
     "localized_name": "string_b",
     "name": "string_b",
     "type": "STRING",
     "link": 10,
     "widget": {
      "name": "string_b"
     }
    }
   ],
   "outputs": [
    {
     "localized_name": "STRING",
     "name": "STRING",
     "type": "STRING",
     "slot_index": 0,
     "links": [
      12
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "StringConcatenate",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "",
    "",
    " "
   ]
  },
  {
   "id": 17,
   "type": "CLIPTextEncode",
   "pos": [
    400,
    900
   ],
   "size": [
    300,
    90
   ],
   "flags": {},
   "order": 17,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "clip",
     "name": "clip",
     "type": "CLIP",
     "link": 11
    },
    {
     "localized_name": "text",
     "name": "text",
     "type": "STRING",
     "link": 12,
     "widget": {
      "name": "text"
     }
    }
   ],
   "outputs": [
    {
     "localized_name": "CONDITIONING",
     "name": "CONDITIONING",
     "type": "CONDITIONING",
     "slot_index": 0,
     "links": [
      13,
      23
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "CLIPTextEncode",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    ""
   ],
   "title": "Prompt (positiv)"
  },
  {
   "id": 18,
   "type": "ConditioningZeroOut",
   "pos": [
    400,
    1060
   ],
   "size": [
    240,
    30
   ],
   "flags": {},
   "order": 18,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "conditioning",
     "name": "conditioning",
     "type": "CONDITIONING",
     "link": 13
    }
   ],
   "outputs": [
    {
     "localized_name": "CONDITIONING",
     "name": "CONDITIONING",
     "type": "CONDITIONING",
     "slot_index": 0,
     "links": [
      24
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ConditioningZeroOut",
    "cnr_id": "comfy-core"
   },
   "widgets_values": []
  },
  {
   "id": 19,
   "type": "QwenImageDiffsynthControlnet",
   "pos": [
    720,
    10
   ],
   "size": [
    320,
    160
   ],
   "flags": {},
   "order": 19,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "model",
     "name": "model",
     "type": "MODEL",
     "link": 14
    },
    {
     "localized_name": "model_patch",
     "name": "model_patch",
     "type": "MODEL_PATCH",
     "link": 15
    },
    {
     "localized_name": "vae",
     "name": "vae",
     "type": "VAE",
     "link": 16
    },
    {
     "localized_name": "image",
     "name": "image",
     "type": "IMAGE",
     "link": 17
    },
    {
     "localized_name": "mask",
     "name": "mask",
     "type": "MASK",
     "link": null
    }
   ],
   "outputs": [
    {
     "localized_name": "MODEL",
     "name": "MODEL",
     "type": "MODEL",
     "slot_index": 0,
     "links": [
      18
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "QwenImageDiffsynthControlnet",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    0.75,
    0.0,
    0.9
   ],
   "title": "Skizze → Geometrie (ControlNet)"
  },
  {
   "id": 20,
   "type": "ModelSamplingAuraFlow",
   "pos": [
    720,
    210
   ],
   "size": [
    320,
    58
   ],
   "flags": {},
   "order": 20,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "model",
     "name": "model",
     "type": "MODEL",
     "link": 18
    }
   ],
   "outputs": [
    {
     "localized_name": "MODEL",
     "name": "MODEL",
     "type": "MODEL",
     "slot_index": 0,
     "links": [
      22
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ModelSamplingAuraFlow",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    3
   ]
  },
  {
   "id": 21,
   "type": "GetImageSize",
   "pos": [
    720,
    310
   ],
   "size": [
    220,
    66
   ],
   "flags": {},
   "order": 21,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "image",
     "name": "image",
     "type": "IMAGE",
     "link": 19
    }
   ],
   "outputs": [
    {
     "localized_name": "width",
     "name": "width",
     "type": "INT",
     "slot_index": 0,
     "links": [
      20
     ]
    },
    {
     "localized_name": "height",
     "name": "height",
     "type": "INT",
     "slot_index": 1,
     "links": [
      21
     ]
    },
    {
     "localized_name": "batch_size",
     "name": "batch_size",
     "type": "INT",
     "slot_index": 2,
     "links": null
    }
   ],
   "properties": {
    "Node name for S&R": "GetImageSize",
    "cnr_id": "comfy-core"
   },
   "widgets_values": []
  },
  {
   "id": 22,
   "type": "EmptySD3LatentImage",
   "pos": [
    720,
    420
   ],
   "size": [
    320,
    106
   ],
   "flags": {},
   "order": 22,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "width",
     "name": "width",
     "type": "INT",
     "link": 20,
     "widget": {
      "name": "width"
     }
    },
    {
     "localized_name": "height",
     "name": "height",
     "type": "INT",
     "link": 21,
     "widget": {
      "name": "height"
     }
    }
   ],
   "outputs": [
    {
     "localized_name": "LATENT",
     "name": "LATENT",
     "type": "LATENT",
     "slot_index": 0,
     "links": [
      25
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "EmptySD3LatentImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    1024,
    1024,
    4
   ],
   "title": "Bildgrösse & Anzahl (batch_size)"
  },
  {
   "id": 23,
   "type": "KSampler",
   "pos": [
    720,
    580
   ],
   "size": [
    320,
    262
   ],
   "flags": {},
   "order": 23,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "model",
     "name": "model",
     "type": "MODEL",
     "link": 22
    },
    {
     "localized_name": "positive",
     "name": "positive",
     "type": "CONDITIONING",
     "link": 23
    },
    {
     "localized_name": "negative",
     "name": "negative",
     "type": "CONDITIONING",
     "link": 24
    },
    {
     "localized_name": "latent_image",
     "name": "latent_image",
     "type": "LATENT",
     "link": 25
    }
   ],
   "outputs": [
    {
     "localized_name": "LATENT",
     "name": "LATENT",
     "type": "LATENT",
     "slot_index": 0,
     "links": [
      26
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "KSampler",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    20261019,
    "randomize",
    8,
    1,
    "res_multistep",
    "simple",
    1.0
   ]
  },
  {
   "id": 24,
   "type": "VAEDecode",
   "pos": [
    720,
    880
   ],
   "size": [
    200,
    46
   ],
   "flags": {},
   "order": 24,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "samples",
     "name": "samples",
     "type": "LATENT",
     "link": 26
    },
    {
     "localized_name": "vae",
     "name": "vae",
     "type": "VAE",
     "link": 27
    }
   ],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      28
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "VAEDecode",
    "cnr_id": "comfy-core"
   },
   "widgets_values": []
  },
  {
   "id": 25,
   "type": "SaveImage",
   "pos": [
    1080,
    10
   ],
   "size": [
    580,
    1340
   ],
   "flags": {},
   "order": 25,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "images",
     "name": "images",
     "type": "IMAGE",
     "link": 28
    }
   ],
   "outputs": [],
   "properties": {
    "Node name for S&R": "SaveImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "A_skizze/variante"
   ]
  }
 ],
 "links": [
  [
   1,
   6,
   0,
   7,
   0,
   "IMAGE"
  ],
  [
   2,
   7,
   0,
   8,
   0,
   "IMAGE"
  ],
  [
   3,
   7,
   0,
   9,
   0,
   "IMAGE"
  ],
  [
   4,
   8,
   0,
   10,
   0,
   "IMAGE"
  ],
  [
   5,
   9,
   0,
   10,
   1,
   "IMAGE"
  ],
  [
   6,
   10,
   0,
   11,
   0,
   "IMAGE"
  ],
  [
   7,
   12,
   0,
   15,
   0,
   "STRING"
  ],
  [
   8,
   13,
   0,
   15,
   1,
   "STRING"
  ],
  [
   9,
   15,
   0,
   16,
   0,
   "STRING"
  ],
  [
   10,
   14,
   0,
   16,
   1,
   "STRING"
  ],
  [
   11,
   3,
   0,
   17,
   0,
   "CLIP"
  ],
  [
   12,
   16,
   0,
   17,
   1,
   "STRING"
  ],
  [
   13,
   17,
   0,
   18,
   0,
   "CONDITIONING"
  ],
  [
   14,
   2,
   0,
   19,
   0,
   "MODEL"
  ],
  [
   15,
   5,
   0,
   19,
   1,
   "MODEL_PATCH"
  ],
  [
   16,
   4,
   0,
   19,
   2,
   "VAE"
  ],
  [
   17,
   10,
   0,
   19,
   3,
   "IMAGE"
  ],
  [
   18,
   19,
   0,
   20,
   0,
   "MODEL"
  ],
  [
   19,
   7,
   0,
   21,
   0,
   "IMAGE"
  ],
  [
   20,
   21,
   0,
   22,
   0,
   "INT"
  ],
  [
   21,
   21,
   1,
   22,
   1,
   "INT"
  ],
  [
   22,
   20,
   0,
   23,
   0,
   "MODEL"
  ],
  [
   23,
   17,
   0,
   23,
   1,
   "CONDITIONING"
  ],
  [
   24,
   18,
   0,
   23,
   2,
   "CONDITIONING"
  ],
  [
   25,
   22,
   0,
   23,
   3,
   "LATENT"
  ],
  [
   26,
   23,
   0,
   24,
   0,
   "LATENT"
  ],
  [
   27,
   4,
   0,
   24,
   1,
   "VAE"
  ],
  [
   28,
   24,
   0,
   25,
   0,
   "IMAGE"
  ]
 ],
 "groups": [
  {
   "id": 1,
   "title": "Modelle",
   "bounding": [
    -1000,
    640,
    440,
    470
   ],
   "color": "#555555",
   "font_size": 24,
   "flags": {}
  },
  {
   "id": 2,
   "title": "1 · Skizze vorbereiten",
   "bounding": [
    -400,
    -40,
    1060,
    760
   ],
   "color": "#3f789e",
   "font_size": 24,
   "flags": {}
  },
  {
   "id": 3,
   "title": "2 · Prompt",
   "bounding": [
    -400,
    740,
    1060,
    640
   ],
   "color": "#3f6e3f",
   "font_size": 24,
   "flags": {}
  },
  {
   "id": 4,
   "title": "3 · Generieren",
   "bounding": [
    700,
    -40,
    980,
    1420
   ],
   "color": "#6b4f8a",
   "font_size": 24,
   "flags": {}
  }
 ],
 "config": {},
 "extra": {
  "ds": {
   "scale": 0.6,
   "offset": [
    0,
    0
   ]
  }
 },
 "version": 0.4
}
JSONEOF
  cat > "${WORKFLOW_DIR}/B_Skizze_plus_Materialbilder.json" <<'JSONEOF'
{
 "id": "00000000-0000-4000-8000-000000000006",
 "revision": 0,
 "last_node_id": 16,
 "last_link_id": 17,
 "nodes": [
  {
   "id": 1,
   "type": "MarkdownNote",
   "pos": [
    -1000,
    -40
   ],
   "size": [
    560,
    760
   ],
   "flags": {},
   "order": 1,
   "mode": 0,
   "inputs": [],
   "outputs": [],
   "properties": {},
   "widgets_values": [
    "## B · Skizze + Materialbilder\n**FLUX.3 Image Klein (Multi-Referenz)**\n\nHier steuerst du Materialien mit **Fotos** statt nur mit Text.\n\n1. **Referenz 0 = Skizze** (bestimmt Geometrie, Ausschnitt und Bildformat).\n2. **Referenz 1 / 2 = Materialfotos** – z. B. Beton-Musterplatte, Holz, Naturstein, Metall. Am besten frontal, gleichmässig belichtet, nur das Material im Bild.\n3. Im Prompt sprichst du die Bilder mit `<ref_image_0>`, `<ref_image_1>`, `<ref_image_2>` … an – in der Reihenfolge, in der sie am Node *Referenzen* hängen.\n4. **Optional Stimmungsreferenz**: Den Node *Stimmung (optional)* mit dem freien Eingang `images.image3` am Node *Referenzen* verbinden und im Prompt z. B. schreiben: *\"Match the light, colour grading and atmosphere of <ref_image_3>, but not its content.\"*\n5. **Run** → 4 Varianten. Bestes Bild in **C · Upscale** weiterverarbeiten.\n\n**Material entfernen**: Verbindung zum Node *Referenzen* lösen und den Satz im Prompt löschen. **Mehr Materialien**: einfach weitere Load-Image-Nodes an die freien Eingänge hängen (bis 10 Referenzen).\n\n**Tipps**\n- Prompt als vollständige Bildbeschreibung schreiben, nicht als Befehl.\n- Wenn die Skizze zu frei interpretiert wird: im Prompt betonen *\"keep the exact geometry … of <ref_image_0>\"* und die Skizze sauberer/kontrastreicher machen.\n- `megapixels` am Node *Referenzen* = Ausgabegrösse (1.5 ≈ 1500×1000). Höher = mehr Materialdetail, aber langsamer.\n\nSteps 16 · CFG 1 · euler/simple (destilliertes Modell)."
   ],
   "color": "#223",
   "bgcolor": "#335"
  },
  {
   "id": 2,
   "type": "UNETLoader",
   "pos": [
    -980,
    810
   ],
   "size": [
    360,
    82
   ],
   "flags": {},
   "order": 2,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "MODEL",
     "name": "MODEL",
     "type": "MODEL",
     "slot_index": 0,
     "links": [
      11
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "UNETLoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "flux3-image-klein-bf16.safetensors",
    "default"
   ]
  },
  {
   "id": 3,
   "type": "CLIPLoader",
   "pos": [
    -980,
    910
   ],
   "size": [
    360,
    106
   ],
   "flags": {},
   "order": 3,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "CLIP",
     "name": "CLIP",
     "type": "CLIP",
     "slot_index": 0,
     "links": [
      1
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "CLIPLoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "qwen3vl_4b_int8_convrot.safetensors",
    "flux3",
    "default"
   ]
  },
  {
   "id": 4,
   "type": "VAELoader",
   "pos": [
    -980,
    1030
   ],
   "size": [
    360,
    58
   ],
   "flags": {},
   "order": 4,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "VAE",
     "name": "VAE",
     "type": "VAE",
     "slot_index": 0,
     "links": [
      6,
      16
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "VAELoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "flux2-vae.safetensors"
   ]
  },
  {
   "id": 5,
   "type": "LoadImage",
   "pos": [
    -380,
    10
   ],
   "size": [
    320,
    380
   ],
   "flags": {},
   "order": 5,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      7
     ]
    },
    {
     "localized_name": "MASK",
     "name": "MASK",
     "type": "MASK",
     "slot_index": 1,
     "links": null
    }
   ],
   "properties": {
    "Node name for S&R": "LoadImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "skizze_beispiel.png",
    "image"
   ],
   "title": "Referenz 0 · Skizze"
  },
  {
   "id": 6,
   "type": "LoadImage",
   "pos": [
    -40,
    10
   ],
   "size": [
    320,
    380
   ],
   "flags": {},
   "order": 6,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      8
     ]
    },
    {
     "localized_name": "MASK",
     "name": "MASK",
     "type": "MASK",
     "slot_index": 1,
     "links": null
    }
   ],
   "properties": {
    "Node name for S&R": "LoadImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "material_1.png",
    "image"
   ],
   "title": "Referenz 1 · Material Wand"
  },
  {
   "id": 7,
   "type": "LoadImage",
   "pos": [
    300,
    10
   ],
   "size": [
    320,
    380
   ],
   "flags": {},
   "order": 7,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      9
     ]
    },
    {
     "localized_name": "MASK",
     "name": "MASK",
     "type": "MASK",
     "slot_index": 1,
     "links": null
    }
   ],
   "properties": {
    "Node name for S&R": "LoadImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "material_2.png",
    "image"
   ],
   "title": "Referenz 2 · Material Rahmen/Kante"
  },
  {
   "id": 8,
   "type": "LoadImage",
   "pos": [
    640,
    10
   ],
   "size": [
    320,
    380
   ],
   "flags": {},
   "order": 8,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": null
    },
    {
     "localized_name": "MASK",
     "name": "MASK",
     "type": "MASK",
     "slot_index": 1,
     "links": null
    }
   ],
   "properties": {
    "Node name for S&R": "LoadImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "stimmung.png",
    "image"
   ],
   "title": "Stimmung (optional, nicht verbunden)"
  },
  {
   "id": 9,
   "type": "PrimitiveStringMultiline",
   "pos": [
    -380,
    510
   ],
   "size": [
    760,
    340
   ],
   "flags": {},
   "order": 9,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "STRING",
     "name": "STRING",
     "type": "STRING",
     "slot_index": 0,
     "links": [
      2
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "PrimitiveStringMultiline",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "Create a photorealistic close-up architectural photograph of the building fragment drawn in <ref_image_0>. Keep the exact geometry, proportions, edges, openings and camera angle of the sketch; only a part of the building is visible, tightly cropped. Render the main wall surfaces in the material shown in <ref_image_1>, matching its colour, texture, grain, joint pattern and scale. Render the frames, edges and lintel in the material shown in <ref_image_2>. Medium format camera, 80mm lens, soft overcast late-afternoon light grazing across the surfaces, deep soft shadows inside the reveal, muted desaturated palette, quiet contemplative atmosphere, light haze, subtle film grain. Precise physical material detail. No pencil lines, paper texture or drawing marks remain."
   ],
   "title": "Prompt (Bildbeschreibung mit <ref_image_N>)",
   "color": "#232",
   "bgcolor": "#353"
  },
  {
   "id": 10,
   "type": "CLIPTextEncode",
   "pos": [
    420,
    520
   ],
   "size": [
    300,
    90
   ],
   "flags": {},
   "order": 10,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "clip",
     "name": "clip",
     "type": "CLIP",
     "link": 1
    },
    {
     "localized_name": "text",
     "name": "text",
     "type": "STRING",
     "link": 2,
     "widget": {
      "name": "text"
     }
    }
   ],
   "outputs": [
    {
     "localized_name": "CONDITIONING",
     "name": "CONDITIONING",
     "type": "CONDITIONING",
     "slot_index": 0,
     "links": [
      3,
      4
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "CLIPTextEncode",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    ""
   ],
   "title": "Prompt (positiv)"
  },
  {
   "id": 11,
   "type": "ConditioningZeroOut",
   "pos": [
    420,
    660
   ],
   "size": [
    240,
    30
   ],
   "flags": {},
   "order": 11,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "conditioning",
     "name": "conditioning",
     "type": "CONDITIONING",
     "link": 3
    }
   ],
   "outputs": [
    {
     "localized_name": "CONDITIONING",
     "name": "CONDITIONING",
     "type": "CONDITIONING",
     "slot_index": 0,
     "links": [
      5
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ConditioningZeroOut",
    "cnr_id": "comfy-core"
   },
   "widgets_values": []
  },
  {
   "id": 12,
   "type": "ReferenceMultiLatent",
   "pos": [
    1060,
    10
   ],
   "size": [
    340,
    230
   ],
   "flags": {},
   "order": 12,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "positive",
     "name": "positive",
     "type": "CONDITIONING",
     "link": 4
    },
    {
     "localized_name": "negative",
     "name": "negative",
     "type": "CONDITIONING",
     "link": 5
    },
    {
     "localized_name": "vae",
     "name": "vae",
     "type": "VAE",
     "link": 6
    },
    {
     "localized_name": "images.image0",
     "name": "images.image0",
     "type": "IMAGE",
     "link": 7
    },
    {
     "localized_name": "images.image1",
     "name": "images.image1",
     "type": "IMAGE",
     "link": 8
    },
    {
     "localized_name": "images.image2",
     "name": "images.image2",
     "type": "IMAGE",
     "link": 9
    },
    {
     "localized_name": "images.image3",
     "name": "images.image3",
     "type": "IMAGE",
     "link": null
    }
   ],
   "outputs": [
    {
     "localized_name": "positive",
     "name": "positive",
     "type": "CONDITIONING",
     "slot_index": 0,
     "links": [
      12
     ]
    },
    {
     "localized_name": "negative",
     "name": "negative",
     "type": "CONDITIONING",
     "slot_index": 1,
     "links": [
      13
     ]
    },
    {
     "localized_name": "empty_latent",
     "name": "empty_latent",
     "type": "LATENT",
     "slot_index": 2,
     "links": [
      10
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ReferenceMultiLatent",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "resize",
    1.5
   ],
   "title": "Referenzen"
  },
  {
   "id": 13,
   "type": "RepeatLatentBatch",
   "pos": [
    1060,
    290
   ],
   "size": [
    340,
    58
   ],
   "flags": {},
   "order": 13,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "samples",
     "name": "samples",
     "type": "LATENT",
     "link": 10
    }
   ],
   "outputs": [
    {
     "localized_name": "LATENT",
     "name": "LATENT",
     "type": "LATENT",
     "slot_index": 0,
     "links": [
      14
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "RepeatLatentBatch",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    4
   ],
   "title": "Anzahl Varianten"
  },
  {
   "id": 14,
   "type": "KSampler",
   "pos": [
    1060,
    400
   ],
   "size": [
    320,
    262
   ],
   "flags": {},
   "order": 14,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "model",
     "name": "model",
     "type": "MODEL",
     "link": 11
    },
    {
     "localized_name": "positive",
     "name": "positive",
     "type": "CONDITIONING",
     "link": 12
    },
    {
     "localized_name": "negative",
     "name": "negative",
     "type": "CONDITIONING",
     "link": 13
    },
    {
     "localized_name": "latent_image",
     "name": "latent_image",
     "type": "LATENT",
     "link": 14
    }
   ],
   "outputs": [
    {
     "localized_name": "LATENT",
     "name": "LATENT",
     "type": "LATENT",
     "slot_index": 0,
     "links": [
      15
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "KSampler",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    20261019,
    "randomize",
    16,
    1,
    "euler",
    "simple",
    1.0
   ]
  },
  {
   "id": 15,
   "type": "VAEDecode",
   "pos": [
    1060,
    700
   ],
   "size": [
    200,
    46
   ],
   "flags": {},
   "order": 15,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "samples",
     "name": "samples",
     "type": "LATENT",
     "link": 15
    },
    {
     "localized_name": "vae",
     "name": "vae",
     "type": "VAE",
     "link": 16
    }
   ],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      17
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "VAEDecode",
    "cnr_id": "comfy-core"
   },
   "widgets_values": []
  },
  {
   "id": 16,
   "type": "SaveImage",
   "pos": [
    1440,
    10
   ],
   "size": [
    680,
    1220
   ],
   "flags": {},
   "order": 16,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "images",
     "name": "images",
     "type": "IMAGE",
     "link": 17
    }
   ],
   "outputs": [],
   "properties": {
    "Node name for S&R": "SaveImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "B_material/variante"
   ]
  }
 ],
 "links": [
  [
   1,
   3,
   0,
   10,
   0,
   "CLIP"
  ],
  [
   2,
   9,
   0,
   10,
   1,
   "STRING"
  ],
  [
   3,
   10,
   0,
   11,
   0,
   "CONDITIONING"
  ],
  [
   4,
   10,
   0,
   12,
   0,
   "CONDITIONING"
  ],
  [
   5,
   11,
   0,
   12,
   1,
   "CONDITIONING"
  ],
  [
   6,
   4,
   0,
   12,
   2,
   "VAE"
  ],
  [
   7,
   5,
   0,
   12,
   3,
   "IMAGE"
  ],
  [
   8,
   6,
   0,
   12,
   4,
   "IMAGE"
  ],
  [
   9,
   7,
   0,
   12,
   5,
   "IMAGE"
  ],
  [
   10,
   12,
   2,
   13,
   0,
   "LATENT"
  ],
  [
   11,
   2,
   0,
   14,
   0,
   "MODEL"
  ],
  [
   12,
   12,
   0,
   14,
   1,
   "CONDITIONING"
  ],
  [
   13,
   12,
   1,
   14,
   2,
   "CONDITIONING"
  ],
  [
   14,
   13,
   0,
   14,
   3,
   "LATENT"
  ],
  [
   15,
   14,
   0,
   15,
   0,
   "LATENT"
  ],
  [
   16,
   4,
   0,
   15,
   1,
   "VAE"
  ],
  [
   17,
   15,
   0,
   16,
   0,
   "IMAGE"
  ]
 ],
 "groups": [
  {
   "id": 1,
   "title": "Modelle",
   "bounding": [
    -1000,
    760,
    440,
    330
   ],
   "color": "#555555",
   "font_size": 24,
   "flags": {}
  },
  {
   "id": 2,
   "title": "1 · Referenzbilder",
   "bounding": [
    -400,
    -40,
    1400,
    470
   ],
   "color": "#3f789e",
   "font_size": 24,
   "flags": {}
  },
  {
   "id": 3,
   "title": "2 · Prompt",
   "bounding": [
    -400,
    460,
    1400,
    430
   ],
   "color": "#3f6e3f",
   "font_size": 24,
   "flags": {}
  },
  {
   "id": 4,
   "title": "3 · Generieren",
   "bounding": [
    1040,
    -40,
    1100,
    1300
   ],
   "color": "#6b4f8a",
   "font_size": 24,
   "flags": {}
  }
 ],
 "config": {},
 "extra": {
  "ds": {
   "scale": 0.6,
   "offset": [
    0,
    0
   ]
  }
 },
 "version": 0.4
}
JSONEOF
  cat > "${WORKFLOW_DIR}/C_Upscale.json" <<'JSONEOF'
{
 "id": "00000000-0000-4000-8000-000000000007",
 "revision": 0,
 "last_node_id": 27,
 "last_link_id": 33,
 "nodes": [
  {
   "id": 1,
   "type": "MarkdownNote",
   "pos": [
    -1000,
    -40
   ],
   "size": [
    560,
    640
   ],
   "flags": {},
   "order": 1,
   "mode": 0,
   "inputs": [],
   "outputs": [],
   "properties": {},
   "widgets_values": [
    "## C · Upscale eines ausgewählten Bildes\n**Stufe 1 – Detail-Pass (Z-Image Turbo):** Bild wird 4× hochgerechnet, auf ~4 MP gebracht und mit wenig Denoise neu „durchgezeichnet“. Das bringt echte neue Materialdetails (Poren, Maserung, Patina).\n**Stufe 2 – SeedVR2 7B:** macht daraus ein scharfes, sauberes Bild in hoher Auflösung (Standard ×2 → ca. 16 MP, z. B. 4900×3260).\n\nSo arbeitest du damit:\n1. Ausgewähltes Bild aus A oder B im Node **Ausgewähltes Bild** hochladen (oder aus dem Output-Ordner per Drag & Drop).\n2. Den **gleichen Prompt** wie bei der Generierung in den grünen Block kopieren (für B ohne die `<ref_image>`-Bezüge reicht eine Kurzfassung).\n3. **Run**. Mit dem Vergleichs-Node siehst du vorher/nachher.\n\n**Regler**\n- **Detail-Denoise** (KSampler Stufe 1): 0.25 = sehr nah am Original, 0.40 = mehr neue Details, aber kleine Formänderungen möglich. **0 = Stufe 1 aus.**\n- **Zielgrösse Stufe 1** (megapixels): 4 ist ein guter Wert. Bei kleinen Ausgangsbildern nicht über 6.\n- **SeedVR2-Faktor** (multiplier): 2 = ca. 16 MP. Bei Speicherfehler (out of memory) auf 1.5 reduzieren.\n- Farbkorrektur *lab* hält die Farben des Originals."
   ],
   "color": "#223",
   "bgcolor": "#335"
  },
  {
   "id": 2,
   "type": "UNETLoader",
   "pos": [
    -980,
    690
   ],
   "size": [
    360,
    82
   ],
   "flags": {},
   "order": 2,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "MODEL",
     "name": "MODEL",
     "type": "MODEL",
     "slot_index": 0,
     "links": [
      9
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "UNETLoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "z_image_turbo_bf16.safetensors",
    "default"
   ]
  },
  {
   "id": 3,
   "type": "CLIPLoader",
   "pos": [
    -980,
    790
   ],
   "size": [
    360,
    106
   ],
   "flags": {},
   "order": 3,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "CLIP",
     "name": "CLIP",
     "type": "CLIP",
     "slot_index": 0,
     "links": [
      1
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "CLIPLoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "qwen_3_4b.safetensors",
    "lumina2",
    "default"
   ]
  },
  {
   "id": 4,
   "type": "VAELoader",
   "pos": [
    -980,
    910
   ],
   "size": [
    360,
    58
   ],
   "flags": {},
   "order": 4,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "VAE",
     "name": "VAE",
     "type": "VAE",
     "slot_index": 0,
     "links": [
      8,
      15
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "VAELoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "ae.safetensors"
   ]
  },
  {
   "id": 5,
   "type": "UpscaleModelLoader",
   "pos": [
    -980,
    990
   ],
   "size": [
    400,
    58
   ],
   "flags": {},
   "order": 5,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "UPSCALE_MODEL",
     "name": "UPSCALE_MODEL",
     "type": "UPSCALE_MODEL",
     "slot_index": 0,
     "links": [
      4
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "UpscaleModelLoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "RealESRGAN_x4plus.safetensors"
   ]
  },
  {
   "id": 6,
   "type": "UNETLoader",
   "pos": [
    -980,
    1070
   ],
   "size": [
    360,
    82
   ],
   "flags": {},
   "order": 6,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "MODEL",
     "name": "MODEL",
     "type": "MODEL",
     "slot_index": 0,
     "links": [
      20,
      22
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "UNETLoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "seedvr2_7b_int8_convrot.safetensors",
    "default"
   ],
   "title": "SeedVR2 Modell"
  },
  {
   "id": 7,
   "type": "VAELoader",
   "pos": [
    -980,
    1150
   ],
   "size": [
    360,
    58
   ],
   "flags": {},
   "order": 7,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "VAE",
     "name": "VAE",
     "type": "VAE",
     "slot_index": 0,
     "links": [
      19,
      27
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "VAELoader",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "seedvr2_ema_vae_fp16.safetensors"
   ],
   "title": "SeedVR2 VAE"
  },
  {
   "id": 8,
   "type": "LoadImage",
   "pos": [
    -380,
    10
   ],
   "size": [
    320,
    380
   ],
   "flags": {},
   "order": 8,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      5,
      31
     ]
    },
    {
     "localized_name": "MASK",
     "name": "MASK",
     "type": "MASK",
     "slot_index": 1,
     "links": null
    }
   ],
   "properties": {
    "Node name for S&R": "LoadImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "ausgewaehlt.png",
    "image"
   ],
   "title": "Ausgewähltes Bild"
  },
  {
   "id": 9,
   "type": "PrimitiveStringMultiline",
   "pos": [
    -380,
    430
   ],
   "size": [
    700,
    300
   ],
   "flags": {},
   "order": 9,
   "mode": 0,
   "inputs": [],
   "outputs": [
    {
     "localized_name": "STRING",
     "name": "STRING",
     "type": "STRING",
     "slot_index": 0,
     "links": [
      2
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "PrimitiveStringMultiline",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "Close-up architectural photograph of a building fragment, exactly as drawn in the sketch: a deep window reveal cut into a thick wall, a slender lintel above, the edge of the opening seen slightly from below. Only a part of the building is visible, cropped tightly, no sky line, no full facade. The wall is board-marked in-situ concrete with a fine timber grain imprint, small air pores, subtle efflorescence and darker weathering streaks below the lintel. The lintel is dark patinated steel with a faint rust bloom. The window is a deep frameless glass pane reflecting a soft grey sky. Shot on a medium format camera, 80mm lens, f/5.6, eye-level tripod shot. Soft overcast late-afternoon light grazing across the surface, revealing every texture, deep soft shadows inside the reveal. Muted, desaturated palette of warm greys and dark umber, quiet, contemplative, almost still-life atmosphere, light haze, subtle film grain. Precise physical material detail, photorealistic architectural photography."
   ],
   "title": "Prompt (gleicher wie bei der Generierung)",
   "color": "#232",
   "bgcolor": "#353"
  },
  {
   "id": 10,
   "type": "CLIPTextEncode",
   "pos": [
    -40,
    760
   ],
   "size": [
    300,
    90
   ],
   "flags": {},
   "order": 10,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "clip",
     "name": "clip",
     "type": "CLIP",
     "link": 1
    },
    {
     "localized_name": "text",
     "name": "text",
     "type": "STRING",
     "link": 2,
     "widget": {
      "name": "text"
     }
    }
   ],
   "outputs": [
    {
     "localized_name": "CONDITIONING",
     "name": "CONDITIONING",
     "type": "CONDITIONING",
     "slot_index": 0,
     "links": [
      3,
      11
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "CLIPTextEncode",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    ""
   ],
   "title": "Prompt (positiv)"
  },
  {
   "id": 11,
   "type": "ConditioningZeroOut",
   "pos": [
    -40,
    880
   ],
   "size": [
    240,
    30
   ],
   "flags": {},
   "order": 11,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "conditioning",
     "name": "conditioning",
     "type": "CONDITIONING",
     "link": 3
    }
   ],
   "outputs": [
    {
     "localized_name": "CONDITIONING",
     "name": "CONDITIONING",
     "type": "CONDITIONING",
     "slot_index": 0,
     "links": [
      12
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ConditioningZeroOut",
    "cnr_id": "comfy-core"
   },
   "widgets_values": []
  },
  {
   "id": 12,
   "type": "ImageUpscaleWithModel",
   "pos": [
    420,
    10
   ],
   "size": [
    340,
    46
   ],
   "flags": {},
   "order": 12,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "upscale_model",
     "name": "upscale_model",
     "type": "UPSCALE_MODEL",
     "link": 4
    },
    {
     "localized_name": "image",
     "name": "image",
     "type": "IMAGE",
     "link": 5
    }
   ],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      6
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ImageUpscaleWithModel",
    "cnr_id": "comfy-core"
   },
   "widgets_values": []
  },
  {
   "id": 13,
   "type": "ImageScaleToTotalPixels",
   "pos": [
    420,
    100
   ],
   "size": [
    300,
    106
   ],
   "flags": {},
   "order": 13,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "image",
     "name": "image",
     "type": "IMAGE",
     "link": 6
    }
   ],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      7
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ImageScaleToTotalPixels",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "lanczos",
    4.0,
    16
   ],
   "title": "Zielgrösse Stufe 1 (megapixels)"
  },
  {
   "id": 14,
   "type": "VAEEncode",
   "pos": [
    420,
    250
   ],
   "size": [
    200,
    46
   ],
   "flags": {},
   "order": 14,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "pixels",
     "name": "pixels",
     "type": "IMAGE",
     "link": 7
    },
    {
     "localized_name": "vae",
     "name": "vae",
     "type": "VAE",
     "link": 8
    }
   ],
   "outputs": [
    {
     "localized_name": "LATENT",
     "name": "LATENT",
     "type": "LATENT",
     "slot_index": 0,
     "links": [
      13
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "VAEEncode",
    "cnr_id": "comfy-core"
   },
   "widgets_values": []
  },
  {
   "id": 15,
   "type": "ModelSamplingAuraFlow",
   "pos": [
    420,
    330
   ],
   "size": [
    340,
    58
   ],
   "flags": {},
   "order": 15,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "model",
     "name": "model",
     "type": "MODEL",
     "link": 9
    }
   ],
   "outputs": [
    {
     "localized_name": "MODEL",
     "name": "MODEL",
     "type": "MODEL",
     "slot_index": 0,
     "links": [
      10
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ModelSamplingAuraFlow",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    3
   ]
  },
  {
   "id": 16,
   "type": "KSampler",
   "pos": [
    420,
    430
   ],
   "size": [
    320,
    262
   ],
   "flags": {},
   "order": 16,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "model",
     "name": "model",
     "type": "MODEL",
     "link": 10
    },
    {
     "localized_name": "positive",
     "name": "positive",
     "type": "CONDITIONING",
     "link": 11
    },
    {
     "localized_name": "negative",
     "name": "negative",
     "type": "CONDITIONING",
     "link": 12
    },
    {
     "localized_name": "latent_image",
     "name": "latent_image",
     "type": "LATENT",
     "link": 13
    }
   ],
   "outputs": [
    {
     "localized_name": "LATENT",
     "name": "LATENT",
     "type": "LATENT",
     "slot_index": 0,
     "links": [
      14
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "KSampler",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    20261019,
    "fixed",
    5,
    1,
    "dpmpp_2m_sde",
    "beta",
    0.3
   ],
   "title": "KSampler Stufe 1 (Detail-Denoise)"
  },
  {
   "id": 17,
   "type": "VAEDecode",
   "pos": [
    420,
    730
   ],
   "size": [
    200,
    46
   ],
   "flags": {},
   "order": 17,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "samples",
     "name": "samples",
     "type": "LATENT",
     "link": 14
    },
    {
     "localized_name": "vae",
     "name": "vae",
     "type": "VAE",
     "link": 15
    }
   ],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      16,
      33
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "VAEDecode",
    "cnr_id": "comfy-core"
   },
   "widgets_values": []
  },
  {
   "id": 18,
   "type": "ResizeImageMaskNode",
   "pos": [
    860,
    10
   ],
   "size": [
    340,
    106
   ],
   "flags": {},
   "order": 18,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "input",
     "name": "input",
     "type": "IMAGE,MASK",
     "link": 16
    }
   ],
   "outputs": [
    {
     "localized_name": "resized",
     "name": "resized",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      17,
      29
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "ResizeImageMaskNode",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "scale by multiplier",
    2.0,
    "lanczos"
   ],
   "title": "SeedVR2-Faktor (multiplier)"
  },
  {
   "id": 19,
   "type": "SeedVR2Preprocess",
   "pos": [
    860,
    160
   ],
   "size": [
    340,
    30
   ],
   "flags": {},
   "order": 19,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "resized_images",
     "name": "resized_images",
     "type": "IMAGE",
     "link": 17
    }
   ],
   "outputs": [
    {
     "localized_name": "images",
     "name": "images",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      18
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "SeedVR2Preprocess",
    "cnr_id": "comfy-core"
   },
   "widgets_values": []
  },
  {
   "id": 20,
   "type": "VAEEncodeTiled",
   "pos": [
    860,
    230
   ],
   "size": [
    340,
    150
   ],
   "flags": {},
   "order": 20,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "pixels",
     "name": "pixels",
     "type": "IMAGE",
     "link": 18
    },
    {
     "localized_name": "vae",
     "name": "vae",
     "type": "VAE",
     "link": 19
    }
   ],
   "outputs": [
    {
     "localized_name": "LATENT",
     "name": "LATENT",
     "type": "LATENT",
     "slot_index": 0,
     "links": [
      21,
      25
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "VAEEncodeTiled",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    512,
    128,
    4096,
    8
   ]
  },
  {
   "id": 21,
   "type": "SeedVR2Conditioning",
   "pos": [
    860,
    420
   ],
   "size": [
    340,
    50
   ],
   "flags": {},
   "order": 21,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "model",
     "name": "model",
     "type": "MODEL",
     "link": 20
    },
    {
     "localized_name": "vae_conditioning",
     "name": "vae_conditioning",
     "type": "LATENT",
     "link": 21
    }
   ],
   "outputs": [
    {
     "localized_name": "positive",
     "name": "positive",
     "type": "CONDITIONING",
     "slot_index": 0,
     "links": [
      23
     ]
    },
    {
     "localized_name": "negative",
     "name": "negative",
     "type": "CONDITIONING",
     "slot_index": 1,
     "links": [
      24
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "SeedVR2Conditioning",
    "cnr_id": "comfy-core"
   },
   "widgets_values": []
  },
  {
   "id": 22,
   "type": "KSampler",
   "pos": [
    860,
    510
   ],
   "size": [
    320,
    262
   ],
   "flags": {},
   "order": 22,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "model",
     "name": "model",
     "type": "MODEL",
     "link": 22
    },
    {
     "localized_name": "positive",
     "name": "positive",
     "type": "CONDITIONING",
     "link": 23
    },
    {
     "localized_name": "negative",
     "name": "negative",
     "type": "CONDITIONING",
     "link": 24
    },
    {
     "localized_name": "latent_image",
     "name": "latent_image",
     "type": "LATENT",
     "link": 25
    }
   ],
   "outputs": [
    {
     "localized_name": "LATENT",
     "name": "LATENT",
     "type": "LATENT",
     "slot_index": 0,
     "links": [
      26
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "KSampler",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    20261019,
    "fixed",
    1,
    1,
    "euler",
    "simple",
    1.0
   ],
   "title": "KSampler SeedVR2 (nicht ändern)"
  },
  {
   "id": 23,
   "type": "VAEDecodeTiled",
   "pos": [
    860,
    800
   ],
   "size": [
    340,
    150
   ],
   "flags": {},
   "order": 23,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "samples",
     "name": "samples",
     "type": "LATENT",
     "link": 26
    },
    {
     "localized_name": "vae",
     "name": "vae",
     "type": "VAE",
     "link": 27
    }
   ],
   "outputs": [
    {
     "localized_name": "IMAGE",
     "name": "IMAGE",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      28
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "VAEDecodeTiled",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    512,
    128,
    4096,
    8
   ]
  },
  {
   "id": 24,
   "type": "SeedVR2PostProcessing",
   "pos": [
    860,
    990
   ],
   "size": [
    340,
    82
   ],
   "flags": {},
   "order": 24,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "images",
     "name": "images",
     "type": "IMAGE",
     "link": 28
    },
    {
     "localized_name": "original_resized_images",
     "name": "original_resized_images",
     "type": "IMAGE",
     "link": 29
    }
   ],
   "outputs": [
    {
     "localized_name": "images",
     "name": "images",
     "type": "IMAGE",
     "slot_index": 0,
     "links": [
      30,
      32
     ]
    }
   ],
   "properties": {
    "Node name for S&R": "SeedVR2PostProcessing",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "lab"
   ]
  },
  {
   "id": 25,
   "type": "SaveImage",
   "pos": [
    1300,
    10
   ],
   "size": [
    620,
    1100
   ],
   "flags": {},
   "order": 25,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "images",
     "name": "images",
     "type": "IMAGE",
     "link": 30
    }
   ],
   "outputs": [],
   "properties": {
    "Node name for S&R": "SaveImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [
    "C_upscale/final"
   ]
  },
  {
   "id": 26,
   "type": "ImageCompare",
   "pos": [
    1940,
    10
   ],
   "size": [
    620,
    700
   ],
   "flags": {},
   "order": 26,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "image_a",
     "name": "image_a",
     "type": "IMAGE",
     "link": 31
    },
    {
     "localized_name": "image_b",
     "name": "image_b",
     "type": "IMAGE",
     "link": 32
    }
   ],
   "outputs": [],
   "properties": {
    "Node name for S&R": "ImageCompare",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [],
   "title": "Vorher / Nachher"
  },
  {
   "id": 27,
   "type": "PreviewImage",
   "pos": [
    1940,
    740
   ],
   "size": [
    620,
    380
   ],
   "flags": {},
   "order": 27,
   "mode": 0,
   "inputs": [
    {
     "localized_name": "images",
     "name": "images",
     "type": "IMAGE",
     "link": 33
    }
   ],
   "outputs": [],
   "properties": {
    "Node name for S&R": "PreviewImage",
    "cnr_id": "comfy-core"
   },
   "widgets_values": [],
   "title": "Zwischenstand nach Stufe 1"
  }
 ],
 "links": [
  [
   1,
   3,
   0,
   10,
   0,
   "CLIP"
  ],
  [
   2,
   9,
   0,
   10,
   1,
   "STRING"
  ],
  [
   3,
   10,
   0,
   11,
   0,
   "CONDITIONING"
  ],
  [
   4,
   5,
   0,
   12,
   0,
   "UPSCALE_MODEL"
  ],
  [
   5,
   8,
   0,
   12,
   1,
   "IMAGE"
  ],
  [
   6,
   12,
   0,
   13,
   0,
   "IMAGE"
  ],
  [
   7,
   13,
   0,
   14,
   0,
   "IMAGE"
  ],
  [
   8,
   4,
   0,
   14,
   1,
   "VAE"
  ],
  [
   9,
   2,
   0,
   15,
   0,
   "MODEL"
  ],
  [
   10,
   15,
   0,
   16,
   0,
   "MODEL"
  ],
  [
   11,
   10,
   0,
   16,
   1,
   "CONDITIONING"
  ],
  [
   12,
   11,
   0,
   16,
   2,
   "CONDITIONING"
  ],
  [
   13,
   14,
   0,
   16,
   3,
   "LATENT"
  ],
  [
   14,
   16,
   0,
   17,
   0,
   "LATENT"
  ],
  [
   15,
   4,
   0,
   17,
   1,
   "VAE"
  ],
  [
   16,
   17,
   0,
   18,
   0,
   "IMAGE"
  ],
  [
   17,
   18,
   0,
   19,
   0,
   "IMAGE"
  ],
  [
   18,
   19,
   0,
   20,
   0,
   "IMAGE"
  ],
  [
   19,
   7,
   0,
   20,
   1,
   "VAE"
  ],
  [
   20,
   6,
   0,
   21,
   0,
   "MODEL"
  ],
  [
   21,
   20,
   0,
   21,
   1,
   "LATENT"
  ],
  [
   22,
   6,
   0,
   22,
   0,
   "MODEL"
  ],
  [
   23,
   21,
   0,
   22,
   1,
   "CONDITIONING"
  ],
  [
   24,
   21,
   1,
   22,
   2,
   "CONDITIONING"
  ],
  [
   25,
   20,
   0,
   22,
   3,
   "LATENT"
  ],
  [
   26,
   22,
   0,
   23,
   0,
   "LATENT"
  ],
  [
   27,
   7,
   0,
   23,
   1,
   "VAE"
  ],
  [
   28,
   23,
   0,
   24,
   0,
   "IMAGE"
  ],
  [
   29,
   18,
   0,
   24,
   1,
   "IMAGE"
  ],
  [
   30,
   24,
   0,
   25,
   0,
   "IMAGE"
  ],
  [
   31,
   8,
   0,
   26,
   0,
   "IMAGE"
  ],
  [
   32,
   24,
   0,
   26,
   1,
   "IMAGE"
  ],
  [
   33,
   17,
   0,
   27,
   0,
   "IMAGE"
  ]
 ],
 "groups": [
  {
   "id": 1,
   "title": "Modelle",
   "bounding": [
    -1000,
    640,
    440,
    560
   ],
   "color": "#555555",
   "font_size": 24,
   "flags": {}
  },
  {
   "id": 2,
   "title": "0 · Eingang & Prompt",
   "bounding": [
    -400,
    -40,
    760,
    1000
   ],
   "color": "#3f789e",
   "font_size": 24,
   "flags": {}
  },
  {
   "id": 3,
   "title": "1 · Detail-Pass (Z-Image)",
   "bounding": [
    400,
    -40,
    400,
    1000
   ],
   "color": "#3f6e3f",
   "font_size": 24,
   "flags": {}
  },
  {
   "id": 4,
   "title": "2 · SeedVR2 Upscale",
   "bounding": [
    840,
    -40,
    400,
    1000
   ],
   "color": "#6b4f8a",
   "font_size": 24,
   "flags": {}
  },
  {
   "id": 5,
   "title": "3 · Ergebnis",
   "bounding": [
    1280,
    -40,
    1300,
    1180
   ],
   "color": "#8a6a3a",
   "font_size": 24,
   "flags": {}
  }
 ],
 "config": {},
 "extra": {
  "ds": {
   "scale": 0.6,
   "offset": [
    0,
    0
   ]
  }
 },
 "version": 0.4
}
JSONEOF
}

write_examples() {
  mkdir -p "$INPUT_DIR"
  log "Beispiel-Eingaben erzeugen in ${INPUT_DIR}"
  python - "$INPUT_DIR" >>"$LOG" 2>&1 <<'PYEOF'
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

PYEOF
}

summary() {
  log "-------------------------------------------------------------"
  log "Fertig. Modelle:"
  local line r p s
  for line in "${MODELS[@]}"; do
    IFS="|" read -r r p s <<< "$line"
    if [[ -s "${MODELS_DIR}/${s}/$(basename "$p")" ]]; then log "  ✓ ${s}/$(basename "$p")"
    elif [[ "$p" == *klein-bf16* && -s "${MODELS_DIR}/${s}/flux3-image-klein-int8-convrot.safetensors" ]]; then
      log "  ✓ ${s}/flux3-image-klein-int8-convrot.safetensors (Ersatz für bf16)"
    else log "  ✗ FEHLT ${s}/$(basename "$p")"; fi
  done
  log "Workflows: A_Skizze_zu_Stimmung · B_Skizze_plus_Materialbilder · C_Upscale"
  log "-------------------------------------------------------------"
}

update_comfyui
write_workflows      # vor den Downloads, damit der Klein-Fallback die Datei findet
write_examples
download_models
summary
exit 0
