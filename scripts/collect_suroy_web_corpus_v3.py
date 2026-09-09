#!/usr/bin/env python3
"""
SUROY public-web collector v3.

Key goals:
- broaden independent public tourism sources;
- drop real-estate/classified/directory noise before it enters the corpus;
- require tourism/Surigao relevance at page/chunk level;
- use per-seed crawl budgets;
- preserve source provenance;
- no login, Facebook automation, CAPTCHA bypass, or access-control evasion.
"""
from __future__ import annotations
import argparse,csv,hashlib,html,json,re,sys,time
import urllib.error,urllib.parse,urllib.request,urllib.robotparser
from collections import defaultdict,deque
from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path

UA="Project-SUROY-Academic-Corpus/3.0"
BLOCKED_HOSTS=("facebook.com","instagram.com","tiktok.com")
SKIP_EXT=(".jpg",".jpeg",".png",".gif",".webp",".svg",".mp4",".mp3",".zip",".rar",".7z",".css",".js",".xml",".rss")
REGION=("siargao","surigao","general luna","dapa","del carmen","pilar","san isidro","santa monica","burgos","san benito","socorro","bucas grande","sohoton","sugba","cloud 9","magpupungko","pacifico","hinatuan","britania","cagwait","bislig")
TOURISM=("tour","travel","tourism","destination","attraction","beach","surf","island","lagoon","cave","falls","river","mangrove","itinerary","hotel","resort","restaurant","transport","ferry","airport","snorkel","diving","kayak","visitor","guide","things to do","heritage","museum")
NEGATIVE=("property for sale","lot for sale","house and lot","real estate","marketing and sales","classified","clan in relationships","sqm beach front","sq.m beach front","for rent gl siargao","property for rent")
BOILER=("cookie policy","privacy policy","all rights reserved","subscribe to our newsletter","accept cookies","send it to socials","learn how your comment data is processed","affiliate note:")

def now():return datetime.now(timezone.utc).isoformat()
def norm(s):return re.sub(r"\s+"," ",html.unescape(s)).strip()
def hsh(s):return hashlib.sha256(norm(s).lower().encode()).hexdigest()
def host(u):return urllib.parse.urlsplit(u).netloc.lower().split(":")[0]
def canon(u):
    p=urllib.parse.urlsplit(u); path=re.sub(r"/+","/",p.path or "/")
    if path!="/" and path.endswith("/"):path=path[:-1]
    q=p.query if any(x in p.query for x in ("start=","id=","view=article")) else ""
    return urllib.parse.urlunsplit((p.scheme.lower(),p.netloc.lower(),path,q,""))

def relevance(text):
    low=text.lower()
    return sum(t in low for t in REGION), sum(t in low for t in TOURISM), sum(t in low for t in NEGATIVE)

def relevant_chunk(text):
    r,t,n=relevance(text)
    if n:return False
    return r>=1 and t>=1

def relevant_link(u,anchor):
    low=(urllib.parse.unquote(u)+" "+anchor).lower()
    if any(x in low for x in NEGATIVE):return False
    return any(x in low for x in REGION) or any(x in low for x in TOURISM)

class Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.skip=0;self.depth=0;self.cur=[];self.blocks=[];self.links=[];self.title=[];self.intitle=False
    def handle_starttag(self,t,attrs):
        t=t.lower()
        if t in ("script","style","noscript","svg","canvas","form"):
            self.skip+=1;return
        if self.skip:return
        if t=="title":self.intitle=True
        if t in ("p","li","h1","h2","h3","h4","blockquote","td"):
            self.depth+=1
            if self.depth==1:self.cur=[]
        if t=="a":
            href=dict(attrs).get("href")
            if href:self.links.append((href,""))
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
            href,a=self.links[-1];self.links[-1]=(href,(a+" "+s).strip())

class Robots:
    def __init__(self):self.c={}
    def allowed(self,u):
        h=host(u)
        if h not in self.c:
            rp=urllib.robotparser.RobotFileParser();ru=f"{urllib.parse.urlsplit(u).scheme}://{h}/robots.txt"
            try:
                req=urllib.request.Request(ru,headers={"User-Agent":UA})
                with urllib.request.urlopen(req,timeout=10) as r:rp.parse(r.read(512000).decode("utf-8","replace").splitlines())
            except Exception:rp=None
            self.c[h]=rp
        rp=self.c[h]
        return True if rp is None else rp.can_fetch(UA,u)

def fetch(u,timeout,maxbytes):
    req=urllib.request.Request(u,headers={"User-Agent":UA,"Accept":"text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.5"})
    with urllib.request.urlopen(req,timeout=timeout) as r:
        ct=r.headers.get_content_type();data=r.read(maxbytes+1)
        if len(data)>maxbytes:raise ValueError("response too large")
        return data,ct

def good_block(b,maxchars):
    b=norm(b);low=b.lower()
    if not 40<=len(b)<=maxchars:return False
    if any(x in low for x in BOILER+NEGATIVE):return False
    letters=sum(c.isalpha() for c in b)
    return letters/max(1,len(b))>=.45

def chunk(blocks,minchars,target,maxchars):
    out=[];cur=[];size=0
    for b in blocks:
        if not good_block(b,maxchars):continue
        b=norm(b)
        if cur and size+len(b)+1>target:
            t=norm(" ".join(cur))
            if len(t)>=minchars and relevant_chunk(t):out.append(t[:maxchars])
            cur=[];size=0
        cur.append(b);size+=len(b)+1
    if cur:
        t=norm(" ".join(cur))
        if len(t)>=minchars and relevant_chunk(t):out.append(t[:maxchars])
    return out

def extract(data,u,ct,minchars,target,maxchars):
    if ct=="application/pdf" or u.lower().endswith(".pdf"):
        import fitz
        d=fitz.open(stream=data,filetype="pdf")
        blocks=[norm(p.get_text("text")) for p in d if norm(p.get_text("text"))]
        return "",chunk(blocks,minchars,target,maxchars),[]
    x=Extractor();x.feed(data.decode("utf-8","replace"))
    title=norm(" ".join(x.title))[:300]
    links=[]
    for href,a in x.links:
        try:
            v=canon(urllib.parse.urljoin(u,href))
            if v.startswith(("http://","https://")):links.append((v,norm(a)[:250]))
        except Exception:pass
    return title,chunk(x.blocks,minchars,target,maxchars),links

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sources",default="resources/suroy_web_sources_v3.csv")
    ap.add_argument("--output-dir",default="data/external/suroy_web_v3")
    ap.add_argument("--max-total-pages",type=int,default=1200)
    ap.add_argument("--target-chars",type=int,default=700)
    ap.add_argument("--min-chars",type=int,default=180)
    ap.add_argument("--max-chars",type=int,default=1600)
    ap.add_argument("--delay",type=float,default=1.0)
    ap.add_argument("--timeout",type=int,default=20)
    ap.add_argument("--max-bytes",type=int,default=8_000_000)
    ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()

    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    docs=out/"web_documents.jsonl";res=out/"source_results.csv";man=out/"collection_manifest.json"
    if not a.overwrite and any(p.exists() for p in (docs,res,man)):raise SystemExit("output exists; use --overwrite intentionally")

    with Path(a.sources).open(encoding="utf-8",newline="") as f:
        seeds=[r for r in csv.DictReader(f) if r.get("allow_fetch","1").lower() in ("1","true","yes")]
    for s in seeds:
        s["crawl_depth"]=int(s["crawl_depth"]);s["max_pages"]=int(s["max_pages"])
    meta={s["source_id"]:s for s in seeds}
    q=deque((canon(s["url"]),0,s["source_id"]) for s in seeds)
    visited=set();seen=set();robots=Robots();last=defaultdict(float);scount=defaultdict(int)
    rows=[];pages=emitted=fail=robots_skip=0;started=now()

    with docs.open("w",encoding="utf-8") as sink:
        while q and pages<a.max_total_pages:
            u,d,sid=q.popleft();u=canon(u);s=meta[sid]
            if u in visited or scount[sid]>=s["max_pages"]:continue
            h=host(u);path=urllib.parse.urlsplit(u).path.lower()
            if any(x in h for x in BLOCKED_HOSTS) or path.endswith(SKIP_EXT):continue
            if any(x in urllib.parse.unquote(u).lower() for x in NEGATIVE):continue
            visited.add(u)
            if not robots.allowed(u):
                robots_skip+=1;rows.append([sid,u,"ROBOTS_SKIPPED",0,0,""]);continue
            wait=a.delay-(time.time()-last[h])
            if wait>0:time.sleep(wait)
            try:
                data,ct=fetch(u,a.timeout,a.max_bytes);last[h]=time.time();scount[sid]+=1;pages+=1
                title,chs,links=extract(data,u,ct,a.min_chars,a.target_chars,a.max_chars)
                n=0
                for i,t in enumerate(chs,1):
                    k=hsh(t)
                    if k in seen:continue
                    seen.add(k);emitted+=1;n+=1
                    sink.write(json.dumps({"document_id":f"WEB3-{emitted:07d}","text":t,"source_id":sid,"source_url":u,"source_domain":h,"source_title":title,"source_type":s["source_type"],"region":s["region"],"category":s["category"],"provenance_type":"real_public_web","synthetic":False,"retrieved_at":now(),"chunk_index":i,"text_sha256":k},ensure_ascii=False)+"\n")
                rows.append([sid,u,"SUCCESS",n,len(data),ct])
                if d<s["crawl_depth"]:
                    for v,anchor in links:
                        if host(v)==h and v not in visited and relevant_link(v,anchor):q.append((v,d+1,sid))
            except urllib.error.HTTPError as e:
                fail+=1;rows.append([sid,u,f"HTTP_{e.code}",0,0,""])
            except Exception as e:
                fail+=1;rows.append([sid,u,f"FAILED_{type(e).__name__}",0,0,str(e)[:180]])
            if pages and pages%25==0:print(f"Pages: {pages} | Documents: {emitted} | Queue: {len(q)}",flush=True)

    with res.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["source_id","url","status","documents_emitted","bytes","detail"]);w.writerows(rows)
    m={"state":"completed","collector_version":3,"started_at":started,"completed_at":now(),"seed_sources":len(seeds),"pages_fetched":pages,"documents_emitted":emitted,"unique_text_hashes":len(seen),"fetch_failures":fail,"robots_skips":robots_skip,"source_page_counts":dict(scount),"filters":{"negative_terms":list(NEGATIVE),"chunk_requires_region_and_tourism_term":True},"methodology":{"public_pages_only":True,"robots_respected":True,"facebook_automated_scraping":False,"rate_limit_seconds_per_host":a.delay,"exact_deduplication":"normalized SHA-256"},"output":str(docs.resolve())}
    man.write_text(json.dumps(m,indent=2),encoding="utf-8");print(json.dumps(m,indent=2))
if __name__=="__main__":main()
