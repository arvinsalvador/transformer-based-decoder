#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,html,json,re,sys,time
import urllib.error,urllib.parse,urllib.request,urllib.robotparser
from collections import defaultdict,deque
from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path
USER_AGENT="Project-SUROY-Academic-Corpus/2.0"
BLOCKED=("facebook.com","instagram.com","tiktok.com")
SKIP=(".jpg",".jpeg",".png",".gif",".webp",".svg",".mp4",".mp3",".zip",".rar",".7z",".css",".js",".xml",".rss")
BOILER=("cookie policy","privacy policy","all rights reserved","subscribe to our newsletter","elevating siargao business discovery with refined simplicity","your trusted local business directory for siargao island areas","send it to socials, chats, or copy the link","learn how your comment data is processed","affiliate note: some booking links are affiliate links")
SECTIONS={
"siargaofinder.com":("/blog",),"www.siargaofinder.com":("/blog",),
"discoversiargao.com":("/travel-guide","/siargao-news/travel","/siargao-news/guide","/forum/destinations"),
"www.discoversiargao.com":("/travel-guide","/siargao-news/travel","/siargao-news/guide","/forum/destinations"),
"travelasiargao.com":("/stories",),"www.travelasiargao.com":("/stories",),
"surigaodelnorte.gov.ph":("/",),"www.surigaodelnorte.gov.ph":("/",),
"surigaodelsur.gov.ph":("/tourism",),"www.surigaodelsur.gov.ph":("/tourism",),
"surigaocity.gov.ph":("/ui/tourism",),"www.surigaocity.gov.ph":("/ui/tourism",),
"tourism.gov.ph":("/destination/caraga",),"www.tourism.gov.ph":("/destination/caraga",)
}
def now(): return datetime.now(timezone.utc).isoformat()
def norm(t): return re.sub(r"\s+"," ",html.unescape(t)).strip()
def key(t): return hashlib.sha256(norm(t).lower().encode()).hexdigest()
def host(u): return urllib.parse.urlsplit(u).netloc.lower().split(":")[0]
def canon(u):
    p=urllib.parse.urlsplit(u); path=re.sub(r"/+","/",p.path or "/")
    if path!="/" and path.endswith("/"): path=path[:-1]
    q=p.query if any(x in p.query for x in ("view=article","id=","start=")) else ""
    return urllib.parse.urlunsplit((p.scheme.lower(),p.netloc.lower(),path,q,""))
def relevant(u,a):
    h=host(u); path=urllib.parse.urlsplit(u).path.lower()
    return any(path.startswith(x) for x in SECTIONS.get(h,()))
class X(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.skip=0; self.depth=0; self.cur=[]; self.blocks=[]; self.links=[]; self.title=[]; self.intitle=False
    def handle_starttag(self,t,attrs):
        t=t.lower()
        if t in ("script","style","noscript","svg","canvas","form"): self.skip+=1; return
        if self.skip:return
        if t=="title": self.intitle=True
        if t in ("p","li","h1","h2","h3","h4","blockquote","td"):
            self.depth+=1
            if self.depth==1:self.cur=[]
        if t=="a":
            h=dict(attrs).get("href")
            if h:self.links.append((h,""))
    def handle_endtag(self,t):
        t=t.lower()
        if t in ("script","style","noscript","svg","canvas","form"):
            if self.skip:self.skip-=1
            return
        if self.skip:return
        if t=="title":self.intitle=False
        if t in ("p","li","h1","h2","h3","h4","blockquote","td") and self.depth:
            self.depth-=1
            if self.depth==0:
                s=norm(" ".join(self.cur))
                if s:self.blocks.append(s)
                self.cur=[]
    def handle_data(self,d):
        if self.skip:return
        s=norm(d)
        if not s:return
        if self.intitle:self.title.append(s)
        if self.depth:self.cur.append(s)
        if self.links:
            h,a=self.links[-1]; self.links[-1]=(h,(a+" "+s).strip())
class Robots:
    def __init__(self):self.c={}
    def ok(self,u):
        h=host(u)
        if h not in self.c:
            rp=urllib.robotparser.RobotFileParser(); ru=f"{urllib.parse.urlsplit(u).scheme}://{h}/robots.txt"
            try:
                req=urllib.request.Request(ru,headers={"User-Agent":USER_AGENT})
                with urllib.request.urlopen(req,timeout=10) as r: rp.parse(r.read(512000).decode("utf-8","replace").splitlines())
            except Exception: rp=None
            self.c[h]=rp
        return True if self.c[h] is None else self.c[h].can_fetch(USER_AGENT,u)
def useful(s,maxc):
    if not 40<=len(s)<=maxc:return False
    low=s.lower()
    if any(x in low for x in BOILER):return False
    return sum(c.isalpha() for c in s)/max(1,len(s))>=.45
def chunks(blocks,minc,target,maxc):
    out=[];cur=[];n=0
    for b in blocks:
        b=norm(b)
        if not useful(b,maxc):continue
        if cur and n+len(b)+1>target:
            s=norm(" ".join(cur))
            if len(s)>=minc:out.append(s[:maxc])
            cur=[];n=0
        cur.append(b);n+=len(b)+1
    if cur:
        s=norm(" ".join(cur))
        if len(s)>=minc:out.append(s[:maxc])
    return out
def fetch(u,timeout,maxb):
    req=urllib.request.Request(u,headers={"User-Agent":USER_AGENT,"Accept":"text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.5"})
    with urllib.request.urlopen(req,timeout=timeout) as r:
        ct=r.headers.get_content_type(); data=r.read(maxb+1)
        if len(data)>maxb: raise ValueError("response too large")
        return data,ct
def extract(data,u,minc,target,maxc,ct):
    if ct=="application/pdf" or u.lower().endswith(".pdf"):
        import fitz
        d=fitz.open(stream=data,filetype="pdf")
        b=[norm(p.get_text("text")) for p in d if norm(p.get_text("text"))]
        return "",chunks(b,minc,target,maxc),[]
    p=X();p.feed(data.decode("utf-8","replace"))
    links=[]
    for href,a in p.links:
        try:
            x=canon(urllib.parse.urljoin(u,href))
            if x.startswith(("http://","https://")):links.append((x,norm(a)[:300]))
        except:pass
    return norm(" ".join(p.title))[:300],chunks(p.blocks,minc,target,maxc),links
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sources",default="resources/suroy_web_sources.csv"); ap.add_argument("--output-dir",default="data/external/suroy_web_v2")
    ap.add_argument("--max-total-pages",type=int,default=800); ap.add_argument("--target-chars",type=int,default=800)
    ap.add_argument("--min-chars",type=int,default=220); ap.add_argument("--max-chars",type=int,default=1800)
    ap.add_argument("--delay",type=float,default=1.0); ap.add_argument("--timeout",type=int,default=20); ap.add_argument("--max-bytes",type=int,default=8000000); ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    docs=out/"web_documents.jsonl"; res=out/"source_results.csv"; man=out/"collection_manifest.json"
    if not a.overwrite and any(p.exists() for p in (docs,res,man)): raise SystemExit("output exists; use --overwrite")
    with Path(a.sources).open(encoding="utf-8",newline="") as f:
        seeds=[r for r in csv.DictReader(f) if r.get("allow_fetch","1") in ("1","true","TRUE","yes")]
    for r in seeds: r["crawl_depth"]=int(r["crawl_depth"]); r["max_pages"]=int(r["max_pages"])
    meta={r["source_id"]:r for r in seeds}; q=deque((canon(r["url"]),0,r["source_id"]) for r in seeds)
    visited=set(); seen=set(); robots=Robots(); last=defaultdict(float); scount=defaultdict(int)
    rows=[]; pages=docs_n=fail=robots_n=0; started=now()
    with docs.open("w",encoding="utf-8") as sink:
        while q and pages<a.max_total_pages:
            u,d,sid=q.popleft(); u=canon(u); s=meta[sid]
            if u in visited or scount[sid]>=s["max_pages"]:continue
            h=host(u)
            if any(x in h for x in BLOCKED) or urllib.parse.urlsplit(u).path.lower().endswith(SKIP):continue
            visited.add(u)
            if not robots.ok(u): robots_n+=1; rows.append([sid,u,"ROBOTS_SKIPPED",0,0,""]); continue
            wait=a.delay-(time.time()-last[h])
            if wait>0:time.sleep(wait)
            try:
                data,ct=fetch(u,a.timeout,a.max_bytes); last[h]=time.time(); scount[sid]+=1; pages+=1
                title,chs,links=extract(data,u,a.min_chars,a.target_chars,a.max_chars,ct)
                n=0
                for i,t in enumerate(chs,1):
                    k=key(t)
                    if k in seen:continue
                    seen.add(k); docs_n+=1; n+=1
                    sink.write(json.dumps({"document_id":f"WEB2-{docs_n:07d}","text":t,"source_id":sid,"source_url":u,"source_domain":h,"source_title":title,"source_type":s["source_type"],"region":s["region"],"category":s["category"],"provenance_type":"real_public_web","synthetic":False,"retrieved_at":now(),"chunk_index":i,"text_sha256":k},ensure_ascii=False)+"\n")
                rows.append([sid,u,"SUCCESS",n,len(data),ct])
                if d<s["crawl_depth"]:
                    for x,anchor in links:
                        if host(x)==h and x not in visited and relevant(x,anchor): q.append((x,d+1,sid))
            except urllib.error.HTTPError as e: fail+=1; rows.append([sid,u,f"HTTP_{e.code}",0,0,""])
            except Exception as e: fail+=1; rows.append([sid,u,f"FAILED_{type(e).__name__}",0,0,str(e)[:180]])
            if pages and pages%25==0: print(f"Pages: {pages} | Documents: {docs_n} | Queue: {len(q)}",flush=True)
    with res.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["source_id","url","status","documents_emitted","bytes","detail"]); w.writerows(rows)
    m={"state":"completed","collector_version":2,"started_at":started,"completed_at":now(),"seed_sources":len(seeds),"pages_fetched":pages,"documents_emitted":docs_n,"unique_text_hashes":len(seen),"fetch_failures":fail,"robots_skips":robots_n,"max_total_pages":a.max_total_pages,"chunk_target_chars":a.target_chars,"source_page_counts":dict(scount),"methodology":{"public_pages_only":True,"robots_respected":True,"facebook_automated_scraping":False,"rate_limit_seconds_per_host":a.delay,"exact_deduplication":"normalized SHA-256"},"output":str(docs.resolve())}
    man.write_text(json.dumps(m,indent=2),encoding="utf-8"); print(json.dumps(m,indent=2))
if __name__=="__main__": main()
