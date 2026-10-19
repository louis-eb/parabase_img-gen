import glob, os, sys
B = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
tpl = open(os.path.join(B, "provision_template.sh"), encoding="utf-8").read()

wf_cmds = []
for f in sorted(glob.glob(os.path.join(OUT, "workflows", "*.json"))):
    name = os.path.basename(f)
    body = open(f, encoding="utf-8").read()
    assert "JSONEOF" not in body
    wf_cmds.append(f'  cat > "${{WORKFLOW_DIR}}/{name}" <<\'JSONEOF\'\n{body}\nJSONEOF')
tpl = tpl.replace("__WORKFLOWS__", "\n".join(wf_cmds))

ex = open(os.path.join(B, "make_examples.py"), encoding="utf-8").read()
assert "PYEOF" not in ex
tpl = tpl.replace("__EXAMPLES__", ex)

dst = os.path.join(OUT, "provisioning_arch.sh")
open(dst, "w", encoding="utf-8", newline="\n").write(tpl)
os.chmod(dst, 0o755)
print("wrote", dst, len(tpl) // 1024, "KB")
