#!/usr/bin/env python3
"""Job runner on the vast.ai instance.

Claude cannot reach the instance directly, so both sides talk through a private
GitHub repo (DATA_REPO):

  branch main     <- Claude writes:  data/, configs/, jobs/<id>.json
  branch results  <- this agent writes: results/<id>/status.json, log.txt, previews

Job types (json field "type"):
  shell   {"cmd": "...", "timeout": 1800}                      run a shell command
  train   {"config": "configs/x.yaml", "name": "x", "prep": "..."}   AI Toolkit run (GPU lane)
  comfy   {"prompt": {<API graph>}, "inputs": ["data/inputs/a.png"]} ComfyUI run (GPU lane)
  export  {"files": ["/workspace/ComfyUI/output/..png"]}       push full-res files
  cancel  {"target": "<job id>"}                               stop a running job
  stop    {}                                                   stop the instance

Full-resolution images stay on the instance; only small JPG previews are pushed.
The agent stops the instance after AUTO_STOP_MIN minutes without jobs, without
training and without ComfyUI activity (0 = never).
"""
import glob
import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request

ROOT = os.environ.get("RELAY_DIR", "/workspace/relay")
MAIN = os.path.join(ROOT, "main")
RES = os.path.join(ROOT, "results")
STATE = os.path.join(ROOT, "state.json")
REPO = os.environ.get("DATA_REPO", "")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
POLL = int(os.environ.get("RELAY_POLL", "20"))
AUTO_STOP_MIN = float(os.environ.get("AUTO_STOP_MIN", "40"))
COMFY_DIR = os.environ.get("COMFYUI_DIR", "/workspace/ComfyUI")
AITK_DIR = os.environ.get("AITK_DIR", "/workspace/ai-toolkit")
AITK_PY = os.environ.get("AITK_PY", "/venv/aitk/bin/python")
PREVIEW_PX = int(os.environ.get("PREVIEW_PX", "1280"))

lock = threading.Lock()          # guards the results worktree
running = {}                     # job id -> Popen (cancellable)
last_activity = time.time()
gpu_q = queue.Queue()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd, cwd=None, check=True, timeout=None):
    r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True,
                       text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd}: {r.stderr[-2000:]}")
    return r.stdout


def url():
    if REPO.startswith(("/", "file:")):  # local test remote
        return REPO
    return f"https://x-access-token:{TOKEN}@github.com/{REPO}.git"


# ------------------------------------------------------------------ git plumbing
def setup_repo():
    os.makedirs(ROOT, exist_ok=True)
    if not os.path.isdir(os.path.join(MAIN, ".git")):
        sh(["git", "clone", "-q", "--branch", "main", url(), MAIN])
    sh(["git", "remote", "set-url", "origin", url()], cwd=MAIN)
    for k, v in (("user.name", "vast-agent"), ("user.email", "agent@localhost")):
        sh(["git", "config", k, v], cwd=MAIN)
    if not os.path.isdir(RES):
        has = sh(["git", "ls-remote", "--heads", "origin", "results"], cwd=MAIN).strip()
        if has:
            sh(["git", "fetch", "-q", "origin", "results"], cwd=MAIN)
            sh(["git", "worktree", "add", "-q", RES, "-B", "results", "origin/results"], cwd=MAIN)
        else:
            sh(["git", "worktree", "add", "-q", "--detach", RES], cwd=MAIN)
            sh("git checkout -q --orphan results && git rm -rq --cached . && git clean -fdxq", cwd=RES)
            with open(os.path.join(RES, "README.md"), "w") as f:
                f.write("Results written by remote/agent.py on the GPU instance.\n")
            sh("git add -A && git commit -qm 'results branch'", cwd=RES)
            sh(["git", "push", "-q", "-u", "origin", "results"], cwd=RES)


def pull_main():
    sh(["git", "fetch", "-q", "origin", "main"], cwd=MAIN)
    sh(["git", "reset", "-q", "--hard", "origin/main"], cwd=MAIN)


def push_results(msg):
    with lock:
        sh(["git", "add", "-A"], cwd=RES)
        if not sh(["git", "status", "--porcelain"], cwd=RES).strip():
            return
        sh(["git", "commit", "-qm", msg], cwd=RES)
        for i in range(4):
            r = subprocess.run(["git", "push", "-q", "origin", "results"], cwd=RES,
                               capture_output=True, text=True)
            if r.returncode == 0:
                return
            time.sleep(5 * (i + 1))
        log("push failed:", r.stderr[-500:])


# ------------------------------------------------------------------ helpers
def rdir(jid):
    d = os.path.join(RES, "results", jid)
    os.makedirs(d, exist_ok=True)
    return d


def status(jid, **kw):
    p = os.path.join(rdir(jid), "status.json")
    s = json.load(open(p)) if os.path.exists(p) else {}
    s.update(kw)
    s["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with lock:
        json.dump(s, open(p, "w"), indent=1)
    return s


def tail(path, n=12000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            txt = f.read().decode("utf-8", "replace")
        return txt.replace("\r", "\n")
    except FileNotFoundError:
        return ""


def preview(src, dst, px=PREVIEW_PX):
    from PIL import Image
    im = Image.open(src).convert("RGB")
    im.thumbnail((px, px))
    im.save(dst, "JPEG", quality=85)


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"done": []}


def save_state(st):
    json.dump(st, open(STATE, "w"))


def comfy_base():
    for port in (18188, 8188):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=3)
            return f"http://127.0.0.1:{port}"
        except Exception:
            pass
    return None


def http_json(u, data=None, timeout=30):
    req = urllib.request.Request(u, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return {"http_error": e.code, **json.loads(body)}
        except ValueError:
            return {"http_error": e.code, "body": body[:3000]}


# ------------------------------------------------------------------ job types
def run_proc(jid, cmd, cwd=None, timeout=None, env=None, on_tick=None, tick=60):
    logf = os.path.join(rdir(jid), "log.txt")
    full = os.path.join(ROOT, "logs", jid + ".log")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as out:
        p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                             env={**os.environ, **(env or {})}, preexec_fn=os.setsid)
        running[jid] = p
        t0, last = time.time(), 0
        while p.poll() is None:
            time.sleep(5)
            if timeout and time.time() - t0 > timeout:
                os.killpg(p.pid, signal.SIGTERM)
            if time.time() - last > tick:
                last = time.time()
                with lock:
                    open(logf, "w").write(tail(full))
                if on_tick:
                    on_tick()
                push_results(f"{jid}: progress")
        running.pop(jid, None)
    with lock:
        open(logf, "w").write(tail(full, 40000))
    return p.returncode


def job_shell(jid, j):
    rc = run_proc(jid, j["cmd"], cwd=j.get("cwd", MAIN), timeout=j.get("timeout", 3600))
    return {"returncode": rc}, rc == 0


def job_train(jid, j):
    cfg = os.path.join(MAIN, j["config"])
    name = j["name"]
    if j.get("prep"):
        rc = run_proc(jid, j["prep"], cwd=MAIN, timeout=3600)
        if rc != 0:
            return {"stage": "prep", "returncode": rc}, False
    outdir = os.path.join(AITK_DIR, "output", name)
    sent = set()

    def tick():
        sd = os.path.join(rdir(jid), "samples")
        os.makedirs(sd, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(outdir, "samples", "*"))):
            b = os.path.basename(f)
            if b not in sent and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                try:
                    preview(f, os.path.join(sd, os.path.splitext(b)[0] + ".jpg"), 768)
                    sent.add(b)
                except Exception:
                    pass
        ck = sorted(os.path.basename(c) for c in glob.glob(os.path.join(outdir, "*.safetensors")))
        status(jid, checkpoints=ck)

    env = {"MODELS_PATH": os.environ.get("AITK_MODELS", "/workspace/aitk_models"),
           "HF_HUB_ENABLE_HF_TRANSFER": "1"}
    rc = run_proc(jid, f'"{AITK_PY}" run.py "{cfg}"', cwd=AITK_DIR, env=env, on_tick=tick, tick=120)
    tick()
    link_models()
    copied = []
    dst = os.path.join(COMFY_DIR, "models", "loras", name)
    os.makedirs(dst, exist_ok=True)
    for c in glob.glob(os.path.join(outdir, "*.safetensors")):
        shutil.copy2(c, dst)
        copied.append(os.path.join("loras", name, os.path.basename(c)))
    return {"returncode": rc, "copied_to_comfy": copied}, rc == 0


# files the toolkit downloaded -> ComfyUI names (saves downloading them twice)
LINKS = {"diffusion_models/flux3_image_klein_bf16.safetensors": "diffusion_models/flux3-image-klein-base-bf16.safetensors",
         "text_encoders/qwen3vl_4b_bf16.safetensors": "text_encoders/qwen3vl_4b_bf16.safetensors",
         "vae/flux2-vae.safetensors": "vae/flux2-vae.safetensors"}


def link_models():
    src_root = os.environ.get("AITK_MODELS", "/workspace/aitk_models")
    for src, dst in LINKS.items():
        s, d = os.path.join(src_root, src), os.path.join(COMFY_DIR, "models", dst)
        if os.path.exists(s) and not os.path.exists(d):
            os.makedirs(os.path.dirname(d), exist_ok=True)
            os.symlink(s, d)
            log("linked", d)


def job_comfy(jid, j):
    link_models()
    base = comfy_base()
    if not base:
        return {"error": "ComfyUI not reachable"}, False
    for rel in j.get("inputs", []):
        shutil.copy2(os.path.join(MAIN, rel), os.path.join(COMFY_DIR, "input", os.path.basename(rel)))
    pids = []
    for _ in range(int(j.get("repeat", 1))):
        r = http_json(base + "/prompt", {"prompt": j["prompt"], "client_id": "relay"})
        if r.get("node_errors") or r.get("http_error") or "prompt_id" not in r:
            return {"comfy_rejected": r}, False
        pids.append(r["prompt_id"])
    outs, t0 = [], time.time()
    pd = rdir(jid)
    for pid in pids:
        while True:
            h = http_json(f"{base}/history/{pid}")
            if pid in h:
                break
            if time.time() - t0 > j.get("timeout", 3600):
                return {"error": "timeout", "images": outs}, False
            time.sleep(3)
        st = h[pid].get("status", {})
        if st.get("status_str") == "error":
            return {"error": st.get("messages", [])[-3:], "images": outs}, False
        for node in h[pid].get("outputs", {}).values():
            for im in node.get("images", []):
                if im.get("type") != "output":
                    continue
                full = os.path.join(COMFY_DIR, "output", im.get("subfolder", ""), im["filename"])
                pv = os.path.splitext(im["filename"])[0] + ".jpg"
                preview(full, os.path.join(pd, pv))
                outs.append({"full": full, "preview": pv})
        status(jid, images=outs)
        push_results(f"{jid}: images")
    return {"images": outs}, True


def job_export(jid, j):
    d = os.path.join(rdir(jid), "full")
    os.makedirs(d, exist_ok=True)
    got = []
    for f in j["files"]:
        for g in glob.glob(f):
            shutil.copy2(g, d)
            got.append(os.path.basename(g))
    return {"files": got}, bool(got)


def job_cancel(jid, j):
    p = running.get(j["target"])
    if p:
        os.killpg(p.pid, signal.SIGTERM)
        return {"cancelled": j["target"]}, True
    return {"error": "not running"}, False


def stop_instance(reason):
    cid = os.environ.get("CONTAINER_ID") or os.environ.get("VAST_CONTAINERLABEL", "").replace("C.", "")
    status("_agent", state="stopping", reason=reason)
    push_results("agent: stopping instance")
    log("stopping instance", cid, reason)
    key = os.environ.get("CONTAINER_API_KEY")
    subprocess.run(f"vastai stop instance {cid}" + (f" --api-key {key}" if key else ""), shell=True)


def job_stop(jid, j):
    status(jid, state="done", result={"stopping": True})
    threading.Thread(target=lambda: (time.sleep(5), stop_instance("stop job")), daemon=True).start()
    return {"stopping": True}, True


CPU = {"shell": job_shell, "export": job_export, "cancel": job_cancel, "stop": job_stop}
GPU = {"train": job_train, "comfy": job_comfy}


def execute(jid, j):
    global last_activity
    fn = CPU.get(j.get("type")) or GPU.get(j.get("type"))
    status(jid, state="running", type=j.get("type"), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    push_results(f"{jid}: started")
    try:
        res, ok = fn(jid, j) if fn else ({"error": f"unknown type {j.get('type')}"}, False)
    except Exception:
        res, ok = {"error": traceback.format_exc()[-3000:]}, False
    status(jid, state="done" if ok else "error", result=res, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
    push_results(f"{jid}: {'done' if ok else 'error'}")
    last_activity = time.time()


def gpu_worker():
    while True:
        jid, j = gpu_q.get()
        execute(jid, j)


# ------------------------------------------------------------------ idle watch
_comfy_seen = {"n": None}


def comfy_busy():
    base = comfy_base()
    if not base:
        return False
    try:
        q = http_json(base + "/queue", timeout=5)
        if q.get("queue_running") or q.get("queue_pending"):
            return True
        n = len(http_json(base + "/history?max_items=200", timeout=10))
        changed = _comfy_seen["n"] is not None and n != _comfy_seen["n"]
        _comfy_seen["n"] = n
        return changed
    except Exception:
        return False


def main():
    global last_activity
    setup_repo()
    threading.Thread(target=gpu_worker, daemon=True).start()
    st = load_state()
    # jobs that already have results (e.g. from an earlier instance) never run again
    for d in glob.glob(os.path.join(RES, "results", "*", "status.json")):
        jid = os.path.basename(os.path.dirname(d))
        if jid not in st["done"]:
            st["done"].append(jid)
    save_state(st)
    status("_agent", state="online", since=time.strftime("%Y-%m-%d %H:%M:%S"),
           gpu=sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False).strip())
    push_results("agent online")
    last_beat = time.time()
    while True:
        try:
            pull_main()
            for f in sorted(glob.glob(os.path.join(MAIN, "jobs", "*.json"))):
                jid = os.path.splitext(os.path.basename(f))[0]
                if jid in st["done"]:
                    continue
                st["done"].append(jid)
                save_state(st)
                j = json.load(open(f))
                last_activity = time.time()
                log("job", jid, j.get("type"))
                if j.get("type") in GPU:
                    status(jid, state="queued")
                    push_results(f"{jid}: queued")
                    gpu_q.put((jid, j))
                else:
                    threading.Thread(target=execute, args=(jid, j), daemon=True).start()
        except Exception:
            log("loop error", traceback.format_exc()[-1500:])
        if running or not gpu_q.empty() or comfy_busy():
            last_activity = time.time()
        idle = (time.time() - last_activity) / 60
        if time.time() - last_beat > 600:
            status("_agent", state="online", idle_min=round(idle, 1))
            push_results("agent heartbeat")
            last_beat = time.time()
        if AUTO_STOP_MIN > 0 and idle > AUTO_STOP_MIN:
            stop_instance(f"idle {idle:.0f} min")
            last_activity = time.time() + 3600  # don't hammer while stopping
        time.sleep(POLL)


if __name__ == "__main__":
    if not REPO or not TOKEN:
        raise SystemExit("DATA_REPO and GITHUB_TOKEN must be set")
    main()
