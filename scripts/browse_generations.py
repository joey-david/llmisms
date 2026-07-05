#!/usr/bin/env python3
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as pq

HTML = """<!doctype html><meta charset="utf-8"><title>LLMism generations</title>
<style>
body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#f7f7f7;color:#222}
select,button{font:inherit;padding:.6rem}select{width:100%}.tabs{display:flex;gap:.5rem;margin:1rem 0}
button[aria-selected=true]{background:#222;color:white}.answer{background:white;padding:1rem;margin:1rem 0;border-radius:8px}
pre{white-space:pre-wrap;font:inherit;line-height:1.45}:focus-visible{outline:3px solid #2684ff}
</style>
<h1>Generation browser</h1><label for="prompt">Prompt</label><select id="prompt"></select>
<div class="tabs" role="tablist" aria-label="Model stage"></div><main id="answers"></main>
<script>
const stages={base:"Base",sft:"SFT",dpo:"DPO",rlvr:"RLVR"}, q=new URLSearchParams(location.search);
let data,stage=q.get("stage")||"sft"; const sel=document.querySelector("#prompt"),tabs=document.querySelector(".tabs");
for(const [id,label] of Object.entries(stages)){let b=document.createElement("button");b.textContent=label;b.dataset.stage=id;b.role="tab";b.onclick=()=>show(id);tabs.append(b)}
fetch("/prompts").then(r=>r.json()).then(ps=>{for(const p of ps)sel.add(new Option(p.text,p.id));sel.value=q.get("prompt")||ps[0].id;sel.onchange=load;load()});
function load(){fetch("/answers?prompt="+encodeURIComponent(sel.value)).then(r=>r.json()).then(x=>{data=x;show(stage)})}
function show(s){stage=s;q.set("prompt",sel.value);q.set("stage",s);history.replaceState(null,"","?"+q);
  for(const b of tabs.children)b.setAttribute("aria-selected",b.dataset.stage===s);
  document.querySelector("#answers").innerHTML=(data[s]||[]).map((a,i)=>`<section class=answer><h2>Generation ${i+1} · seed ${a.seed}</h2><pre></pre></section>`).join("");
  [...document.querySelectorAll("pre")].forEach((e,i)=>e.textContent=data[s][i].text)}
</script>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    rows = pq.read_table(
        args.parquet,
        columns=["prompt_id", "prompt", "stage", "greedy", "base_seed", "text"],
    ).to_pylist()
    prompts = list({row["prompt_id"]: row["prompt"] for row in rows}.items())
    answers = {}
    for row in rows:
        if not row["greedy"]:
            answers.setdefault(row["prompt_id"], {}).setdefault(row["stage"], []).append(
                {"seed": row["base_seed"], "text": row["text"]}
            )
    for by_stage in answers.values():
        for stage_rows in by_stage.values():
            stage_rows.sort(key=lambda row: row["seed"])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            if request.path == "/prompts":
                body = json.dumps(
                    [{"id": key, "text": text} for key, text in prompts]
                ).encode()
                content_type = "application/json"
            elif request.path == "/answers":
                prompt_id = parse_qs(request.query).get("prompt", [""])[0]
                body = json.dumps(answers.get(prompt_id, {})).encode()
                content_type = "application/json"
            else:
                body, content_type = HTML.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    print(f"Open http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
