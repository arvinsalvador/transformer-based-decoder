#!/usr/bin/env python3
"""Merge v1/v2/v3 real-web corpora, drop obvious noise, exact-deduplicate, and optionally near-deduplicate."""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path

NEG=("property for sale","lot for sale","house and lot","real estate","marketing and sales","classified","clan in relationships","sqm beach front","sq.m beach front","for rent gl siargao","property for rent")
def norm(s):return re.sub(r"\s+"," ",s).strip()
def key(s):return hashlib.sha256(norm(s).lower().encode()).hexdigest()
def shingles(s,k=5):
    toks=re.findall(r"[a-z0-9]+",s.lower())
    return set(tuple(toks[i:i+k]) for i in range(max(0,len(toks)-k+1)))
def similar(a,b,threshold):
    if not a or not b:return False
    inter=len(a&b);union=len(a|b)
    return union and inter/union>=threshold

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("inputs",nargs="+")
    ap.add_argument("--output",default="data/external/suroy_web_merged/web_documents.jsonl")
    ap.add_argument("--near-dup-threshold",type=float,default=0.92)
    ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and not a.overwrite:raise SystemExit("output exists; use --overwrite intentionally")
    exact=set();accepted=[];sig=[];seen_in=0;noise=0;dups=0;near=0
    for inp in a.inputs:
        with Path(inp).open(encoding="utf-8") as f:
            for line in f:
                seen_in+=1;r=json.loads(line);text=norm(r.get("text",""))
                low=text.lower()
                if len(text)<180 or any(x in low for x in NEG):
                    noise+=1;continue
                h=key(text)
                if h in exact:
                    dups+=1;continue
                sh=shingles(text)
                # Corpus is small enough for a conservative near-dup pass.
                isnear=False
                for prev in sig:
                    if similar(sh,prev,a.near_dup_threshold):
                        isnear=True;break
                if isnear:
                    near+=1;continue
                exact.add(h);sig.append(sh);accepted.append(r)
    with out.open("w",encoding="utf-8") as f:
        for i,r in enumerate(accepted,1):
            r["document_id"]=f"WEB-MERGED-{i:07d}"
            f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(json.dumps({"input_records":seen_in,"accepted":len(accepted),"noise_filtered":noise,"exact_duplicates":dups,"near_duplicates":near,"output":str(out)},indent=2))
if __name__=="__main__":main()
