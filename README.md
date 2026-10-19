# parabase_img-gen

ComfyUI-Setup für stimmungsvolle, präzise Architektur-Fragmente aus Skizzen – läuft auf vast.ai.

| Workflow | Wofür | Modell |
| --- | --- | --- |
| **A · Skizze → Stimmung** | Geometrie exakt aus der Skizze, Material & Licht per Text, 4 Varianten | Z-Image Turbo + Fun ControlNet Union 2.1 |
| **B · Skizze + Materialbilder** | Materialien (und Stimmung) aus Referenzfotos, 4 Varianten | FLUX.3 Image Klein (Multi-Referenz) |
| **C · Upscale** | Ausgewähltes Bild → ca. 16 MP mit neuen Materialdetails | Z-Image Detail-Pass + SeedVR2 7B |

Nur ComfyUI-Core-Nodes, keine Custom Nodes.

## Start auf vast.ai

1. Auf Hugging Face die Lizenz von [flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) akzeptieren und ein **Read**-Token erstellen.
2. vast.ai → *Templates* → offizielles **ComfyUI**-Template → *Edit*, Umgebungsvariablen:
   ```
   PROVISIONING_SCRIPT=https://raw.githubusercontent.com/louis-eb/parabase_img-gen/main/provisioning_arch.sh
   HF_TOKEN=hf_...
   ```
   Disk ≥ 100 GB, Template **privat** speichern (Token!).
3. GPU mieten: RTX 5090 (32 GB) oder RTX 4090 (24 GB), Internet ↓ ≥ 500 Mbps, RAM ≥ 32 GB.
4. Erster Start 15–30 min (~50 GB Modelle). Log: `/workspace/provisioning_arch.log`. Danach ComfyUI öffnen – Workflows A/B/C liegen im Workflow-Browser.

**Bestehende Instanz** (Terminal in Jupyter):
```bash
curl -fsSL https://raw.githubusercontent.com/louis-eb/parabase_img-gen/main/provisioning_arch.sh -o p.sh
HF_TOKEN=hf_... bash p.sh && supervisorctl restart comfyui
```

## Struktur

- `provisioning_arch.sh` – fertiges Skript (enthält die Workflows und erzeugt Beispiel-Eingaben)
- `workflows/` – die drei Workflows als JSON zum direkten Laden in ComfyUI
- `build/` – Quellen: Workflows ändern in `build_workflows.py`, dann neu bauen:
  ```bash
  python3 build/build_workflows.py workflows
  python3 build/assemble.py .
  ```
  `validate.py` prüft die Workflows gegen die offiziellen ComfyUI-Templates (erwartet einen Clone von `Comfy-Org/workflow_templates` unter `/root/wt`).

Anleitung und Prompt-Bibliothek: siehe Claude-Doc *Architektur-Stimmungen · ComfyUI auf vast.ai*.
