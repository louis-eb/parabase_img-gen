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
__WORKFLOWS__
}

write_examples() {
  mkdir -p "$INPUT_DIR"
  log "Beispiel-Eingaben erzeugen in ${INPUT_DIR}"
  python - "$INPUT_DIR" >>"$LOG" 2>&1 <<'PYEOF'
__EXAMPLES__
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
