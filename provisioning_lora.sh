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

# 1 · ComfyUI, models, workflows A/B/C
log "ComfyUI-Setup (provisioning_arch.sh) …"
curl -fsSL "${RAW}/provisioning_arch.sh" -o /tmp/provisioning_arch.sh && bash /tmp/provisioning_arch.sh

# 2 · Base model in ComfyUI too, to compare LoRAs on base vs distilled
dl() {  # repo path subfolder
  local dest="${COMFYUI_DIR}/models/$3/$(basename "$2")" tmp
  [[ -s "$dest" ]] && return 0
  tmp="$(mktemp -d "${WORKSPACE_DIR}/.dl_XXXX")"
  hf download "$1" "$2" --local-dir "$tmp" >>"$LOG" 2>&1 && mv "${tmp}/$2" "$dest"
  local rc=$?; rm -rf "$tmp"; return $rc
}
if [[ -n "${HF_TOKEN:-}" ]]; then
  dl black-forest-labs/flux-3-image-klein-comfy diffusion_models/flux3-image-klein-base-bf16.safetensors diffusion_models \
    || log "WARNUNG: Klein-Base für ComfyUI nicht geladen"
fi

# 3 · Agent code, HEIC support, vast CLI
mkdir -p "${WORKSPACE_DIR}/relay" "${WORKSPACE_DIR}/datasets" "${WORKSPACE_DIR}/aitk_models"
for f in agent.py prep_dataset.py; do curl -fsSL "${RAW}/remote/${f}" -o "${WORKSPACE_DIR}/relay/${f}"; done
"$MAINPY" -m pip install -q --no-cache-dir pillow-heif vastai >>"$LOG" 2>&1 || log "WARNUNG: pip pillow-heif/vastai"

# 4 · AI Toolkit in its own venv (does not touch ComfyUI's packages)
if [[ ! -d "$AITK_DIR" ]]; then
  log "AI Toolkit klonen …"
  git clone -q https://github.com/ostris/ai-toolkit.git "$AITK_DIR" >>"$LOG" 2>&1
else
  git -C "$AITK_DIR" pull -q --ff-only >>"$LOG" 2>&1
fi
if [[ ! -x "${AITK_VENV}/bin/python" ]]; then
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
"${AITK_VENV}/bin/python" -c "import torch;print('aitk torch',torch.__version__,torch.cuda.is_available())" >>"$LOG" 2>&1

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
