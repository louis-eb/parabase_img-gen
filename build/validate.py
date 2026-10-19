import json,glob,sys,collections
# catalog from official templates: socket inputs (non-widget) and widget count per node type
cat=collections.defaultdict(list)
def walk(o):
    if isinstance(o,dict):
        t=o.get("type")
        if isinstance(t,str) and "widgets_values" in o and "inputs" in o and isinstance(o["inputs"],list):
            socks=tuple((i["name"],i.get("type")) for i in o["inputs"] if not i.get("widget") and not i["name"].startswith("images."))
            wv=o["widgets_values"]; wn=len(wv) if isinstance(wv,list) else None
            outs=tuple((x["name"],x.get("type")) for x in o.get("outputs",[]))
            cat[t].append((socks,wn,outs))
        for v in o.values(): walk(v)
    elif isinstance(o,list):
        for v in o: walk(v)
for f in glob.glob("/root/wt/templates/*.json"):
    try: walk(json.load(open(f)))
    except Exception: pass
ok=True
for f in sorted(glob.glob(sys.argv[1]+"/*.json")):
    d=json.load(open(f)); print("==",f.split("/")[-1], len(d["nodes"]),"nodes",len(d["links"]),"links")
    byid={n["id"]:n for n in d["nodes"]}
    # link consistency
    for lid,a,asl,b,bsl,typ in d["links"]:
        oa=byid[a]["outputs"][asl]; ib=byid[b]["inputs"][bsl]
        if lid not in (oa["links"] or []) or ib["link"]!=lid: print("  LINKREF ERR",lid); ok=False
        ta,tb=oa["type"],ib["type"]
        if not (ta==tb or tb=="*" or ta in tb.split(",") or (ib["name"].startswith("on_") )):
            print("  TYPE MISMATCH",lid,byid[a]["type"],ta,"->",byid[b]["type"],ib["name"],tb); ok=False
    for n in d["nodes"]:
        t=n["type"]
        if t=="MarkdownNote": continue
        socks=tuple((i["name"],i.get("type")) for i in n["inputs"] if not i.get("widget") and not i["name"].startswith("images."))
        wn=len(n.get("widgets_values") or [])
        ex=cat.get(t)
        if not ex: print("  (no template ref)",t); continue
        sock_ok=any(set(socks)<=set(e[0]) or set(e[0])<=set(socks) and not (set(socks)-set(e[0])) for e in ex) or any(set(s for s in socks)<=set(e[0])|{("mask","MASK")} for e in ex)
        wcounts=sorted(set(e[1] for e in ex if e[1] is not None))
        flag="" if sock_ok else "  <-- SOCKETS?"
        wflag="" if (wn in wcounts or (wn==0 and None in [e[1] for e in ex]) or (t=="QwenImageDiffsynthControlnet")) else f"  <-- widgets {wn} vs {wcounts}"
        # required sockets must be linked
        unl=[i["name"] for i in n["inputs"] if i["link"] is None and i["name"] not in ("mask","images.image3","on_true","on_false")]
        if flag or wflag or unl: print(f"  {t}: socks={socks} W={wn}{flag}{wflag} unlinked={unl}"); ok=ok and not flag
print("RESULT", "OK" if ok else "PROBLEMS")
