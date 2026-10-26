#!/bin/bash
# vast.ai provisioning: ComfyUI setup (provisioning_arch.sh) + LoRA training + job agent.
#
# Template environment variables:
#   PROVISIONING_SCRIPT = https://raw.githubusercontent.com/louis-eb/parabase_img-gen/main/provisioning_lora.sh
#   HF_TOKEN            = Hugging Face read token (licences accepted for flux-3-image-klein-comfy
#                         AND flux-3-image-klein-base)
#   GITHUB_TOKEN        = fine-grained token, only for the private data repo, "Contents: read and write"
#   DATA_REPO           = owner/name of the private data repo, e.g. louis-eb/parabase-lora-data
#   AUTO_STOP_MIN       = optional, minutes without activity before the instance stops itself (default 40, 0 = off)
#   SETUP               = optional:
#                           "use"   alle ComfyUI-Modelle für A–E, LoRAs aus dem Release, kein Training-Setup
#                           "train" nur was ein neues LoRA-Training braucht (~35 GB)
#                           "full"  beides
#                           "flux2" (Standard) FLUX.2 [dev] fp8 + Turbo-LoRA + SeedVR2 (Hochskalieren) + Klein mit eigenen LoRAs,
#                                   ohne Z-Image/ControlNet (~80 GB, 100–120 GB Disk reichen)
#   LORA_RELEASE        = optional, Release-Tag im Daten-Repo mit den LoRA-Dateien (Standard loras-v1)
#
# Log: /workspace/provisioning_lora.log
set -uo pipefail
RAW="https://raw.githubusercontent.com/louis-eb/parabase_img-gen/main"
WORKSPACE_DIR="${WORKSPACE:-/workspace}"
COMFYUI_DIR="${WORKSPACE_DIR}/ComfyUI"
AITK_DIR="${WORKSPACE_DIR}/ai-toolkit"
AITK_VENV="/venv/aitk"
LOG="${WORKSPACE_DIR}/provisioning_lora.log"
log() { echo "[lora-provisioning] $*" | tee -a "$LOG"; }
MAINPY="/venv/main/bin/python"; [[ -x "$MAINPY" ]] || MAINPY="$(command -v python3)"

SETUP="${SETUP:-flux2}"
if [[ "$SETUP" == "train" ]]; then export ONLY_MODELS="${ONLY_MODELS:-flux3-image-klein-bf16}"; fi
if [[ "$SETUP" == "flux2" ]]; then export ONLY_MODELS="${ONLY_MODELS:-klein-bf16,qwen3vl,flux2-vae,SeedVR2,RealESRGAN}"; fi

# 1 · ComfyUI, models, workflows A/B/C
log "ComfyUI-Setup (provisioning_arch.sh, SETUP=${SETUP}) …"
curl -fsSL "${RAW}/provisioning_arch.sh" -o /tmp/provisioning_arch.sh && bash /tmp/provisioning_arch.sh

# 2 · Base model in ComfyUI too, to compare LoRAs on base vs distilled
dl() {  # repo path subfolder
  local dest="${COMFYUI_DIR}/models/$3/$(basename "$2")" tmp
  [[ -s "$dest" ]] && return 0
  tmp="$(mktemp -d "${WORKSPACE_DIR}/.dl_XXXX")"
  hf download "$1" "$2" --local-dir "$tmp" >>"$LOG" 2>&1 && mv "${tmp}/$2" "$dest"
  local rc=$?; rm -rf "$tmp"; return $rc
}
if [[ -n "${HF_TOKEN:-}" && "$SETUP" == "full" ]]; then
  dl black-forest-labs/flux-3-image-klein-comfy diffusion_models/flux3-image-klein-base-bf16.safetensors diffusion_models \
    || log "WARNUNG: Klein-Base für ComfyUI nicht geladen"
fi

# 2a · FLUX.2 [dev] (fp8) + Mistral-Textencoder (fp4) + Turbo-LoRA – Standardmodell seit 26.10.2026
if [[ "$SETUP" == "flux2" || "$SETUP" == "full" ]]; then
  log "FLUX.2 [dev] laden (~51 GB) …"
  dl Comfy-Org/flux2-dev split_files/diffusion_models/flux2_dev_fp8mixed.safetensors diffusion_models || log "WARNUNG: FLUX.2 dev"
  dl Comfy-Org/flux2-dev split_files/text_encoders/mistral_3_small_flux2_fp4_mixed.safetensors text_encoders || log "WARNUNG: FLUX.2 Textencoder"
  dl Comfy-Org/flux2-dev split_files/loras/Flux2TurboComfyv2.safetensors loras || log "WARNUNG: FLUX.2 Turbo-LoRA"
fi

# 2b · eigene LoRAs aus dem Release des privaten Daten-Repos
if [[ "$SETUP" != "train" && -n "${GITHUB_TOKEN:-}" && -n "${DATA_REPO:-}" ]]; then
  log "LoRAs aus Release ${LORA_RELEASE:-loras-v1} laden …"
  "$MAINPY" - "${COMFYUI_DIR}/models/loras" >>"$LOG" 2>&1 <<'PY' || log "WARNUNG: LoRA-Download"
import json, os, re, sys, urllib.request
dst = sys.argv[1]; repo = os.environ["DATA_REPO"]; tag = os.environ.get("LORA_RELEASE", "loras-v1")
H = {"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"], "Accept": "application/vnd.github+json"}
rel = json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", headers=H)))
for a in rel["assets"]:
    name = a["name"]; sub = re.sub(r"_\d{9}$", "", name.rsplit(".", 1)[0])
    out = os.path.join(dst, sub, name)
    if os.path.exists(out) and os.path.getsize(out) == a["size"]:
        continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    req = urllib.request.Request(a["url"], headers={**H, "Accept": "application/octet-stream"})
    with urllib.request.urlopen(req) as r, open(out + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    os.replace(out + ".part", out); print("LoRA", sub, name)
PY
fi

# 3 · Agent code, HEIC support, vast CLI
mkdir -p "${WORKSPACE_DIR}/relay" "${WORKSPACE_DIR}/datasets" "${WORKSPACE_DIR}/aitk_models"
for f in agent.py prep_dataset.py; do curl -fsSL "${RAW}/remote/${f}" -o "${WORKSPACE_DIR}/relay/${f}"; done
"$MAINPY" -m pip install -q --no-cache-dir pillow-heif vastai >>"$LOG" 2>&1 || log "WARNUNG: pip pillow-heif/vastai"

# 4 · AI Toolkit in its own venv (does not touch ComfyUI's packages) – nur für Training
if [[ "$SETUP" == "use" ]]; then
  log "SETUP=use: kein AI Toolkit"
elif [[ ! -d "$AITK_DIR" ]]; then
  log "AI Toolkit klonen …"
  git clone -q https://github.com/ostris/ai-toolkit.git "$AITK_DIR" >>"$LOG" 2>&1
else
  git -C "$AITK_DIR" pull -q --ff-only >>"$LOG" 2>&1
fi
if [[ "$SETUP" != "use" && ! -x "${AITK_VENV}/bin/python" ]]; then
  log "AI Toolkit venv + torch …"
  PY=python3.12; command -v $PY >/dev/null || PY=python3
  if command -v uv >/dev/null; then uv venv -q --python "$PY" "$AITK_VENV" >>"$LOG" 2>&1; PIP="uv pip install -q --python ${AITK_VENV}/bin/python"
  else "$PY" -m venv "$AITK_VENV"; PIP="${AITK_VENV}/bin/pip install -q --no-cache-dir"; fi
  CUDA_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -1)
  if [[ "${CUDA_MAJOR:-0}" -ge 13 ]]; then IDX=cu130; TV="torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0"
  else IDX=cu128; TV="torch torchvision torchaudio"; fi
  log "Treiber-CUDA ${CUDA_MAJOR:-?} → torch ${IDX}"
  $PIP $TV --index-url "https://download.pytorch.org/whl/${IDX}" >>"$LOG" 2>&1 || log "FEHLER: torch"
  (cd "$AITK_DIR" && $PIP -r requirements.txt >>"$LOG" 2>&1) || log "FEHLER: ai-toolkit requirements"
fi
[[ "$SETUP" != "use" ]] && "${AITK_VENV}/bin/python" -c "import torch;print('aitk torch',torch.__version__,torch.cuda.is_available())" >>"$LOG" 2>&1

# 5 · Data repo + agent as a supervisor service (restarts after a stop/start)
if [[ -n "${GITHUB_TOKEN:-}" && -n "${DATA_REPO:-}" ]]; then
  cat > /etc/supervisor/conf.d/relay-agent.conf <<EOF
[program:relay-agent]
command=${MAINPY} -u ${WORKSPACE_DIR}/relay/agent.py
directory=${WORKSPACE_DIR}/relay
environment=DATA_REPO="${DATA_REPO}",GITHUB_TOKEN="${GITHUB_TOKEN}",HF_TOKEN="${HF_TOKEN:-}",AUTO_STOP_MIN="${AUTO_STOP_MIN:-40}",AITK_PY="${AITK_VENV}/bin/python",AITK_DIR="${AITK_DIR}",COMFYUI_DIR="${COMFYUI_DIR}"
autostart=true
autorestart=true
startsecs=5
stdout_logfile=${WORKSPACE_DIR}/relay/agent.log
redirect_stderr=true
EOF
  supervisorctl reread >>"$LOG" 2>&1; supervisorctl update >>"$LOG" 2>&1
  log "Agent läuft (Log: ${WORKSPACE_DIR}/relay/agent.log)"
else
  log "GITHUB_TOKEN/DATA_REPO fehlen – Agent nicht gestartet (manuelles Arbeiten geht trotzdem)"
fi
log "fertig."
