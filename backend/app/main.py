from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, time, uuid, threading, numpy as np
from .core.io import load_points
from .core.semantic_ai import infer
from .core.adaptive import aggregate, assign
from .core.motion import estimate_motion
from .core.benchmark import benchmark

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONT=os.path.abspath(os.path.join(ROOT,"..","frontend"))
app=FastAPI(title="ADAPT — Adaptive 2.5D Perception")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
JOBS={}; LAST=None; LOCK=threading.Lock()

@app.get("/")
def home(): return FileResponse(os.path.join(FRONT,"landing.html"))

@app.get("/prototype")
def prototype(): return FileResponse(os.path.join(FRONT,"prototype.html"))
@app.get("/health")
def health(): return {"status":"ok","system":"ADAPT","version":"V4","engine":"RandLA-Net + adaptive grid"}

# Laptop-safe preprocessing. This prevents a huge PCD from freezing the browser/API.
def prepare_points(points, mode):
    pts=np.asarray(points,dtype=np.float32)
    original=len(pts)
    limits={"fast":8000,"balanced":14000,"quality":22000}
    max_points=limits.get(mode,14000)
    if original<=max_points: return pts,original,False
    rng=np.random.default_rng(7)
    # Stratified-ish random sampling across XY bins, preserving scene coverage.
    bins={}
    scale=max(1.0,float(np.ptp(pts[:,:2],axis=0).max())/80.0)
    keys=np.floor(pts[:,:2]/scale).astype(np.int32)
    for i,k in enumerate(map(tuple,keys)): bins.setdefault(k,[]).append(i)
    selected=[]
    per=max(1,max_points//max(1,len(bins)))
    for ids in bins.values():
        take=min(len(ids),per); selected.extend(rng.choice(ids,take,replace=False).tolist())
    if len(selected)<max_points:
        remaining=np.setdiff1d(np.arange(original),np.asarray(selected,dtype=np.int64),assume_unique=False)
        extra=rng.choice(remaining,min(max_points-len(selected),len(remaining)),replace=False)
        selected.extend(extra.tolist())
    selected=np.asarray(selected[:max_points],dtype=np.int64)
    return pts[selected],original,True

def setstage(jid,stage,progress,message,**extra):
    JOBS[jid].update({"stage":stage,"progress":progress,"message":message,**extra})

def worker(jid,path,filename,budget,use_previous_frame):
    global LAST
    try:
        t0=time.perf_counter(); setstage(jid,"loading",8,"Loading point cloud…")
        pts_raw=load_points(path)
        if len(pts_raw)==0: raise ValueError("The point cloud contains no points.")
        setstage(jid,"preprocessing",18,f"Preparing {len(pts_raw):,} points for laptop-safe inference…")
        pts,original,downsampled=prepare_points(pts_raw,budget)
        setstage(jid,"ai",30,f"Running RandLA-Net on {len(pts):,} points…")
        labels,conf,names,groups,ai_ok,model=infer(pts)
        setstage(jid,"importance",62,"Computing uncertainty, proximity, geometry and scene importance…")
        motion,transform=estimate_motion(LAST,pts) if use_previous_frame and LAST is not None else ({},None)
        setstage(jid,"adaptive",76,"Allocating adaptive 5 / 15 / 50 cm cells…")
        tg=time.perf_counter(); cells=assign(aggregate(pts,labels,conf,names,groups),budget,motion); grid_t=time.perf_counter()-tg
        setstage(jid,"benchmark",90,"Building benchmark and explainability layers…")
        rc={"0.05":0,"0.15":0,"0.50":0}; sem={}
        for c in cells:
            rc[f'{c["resolution_m"]:.2f}']+=1; sem[c["semantic_class"]]=sem.get(c["semantic_class"],0)+1
        elev=[c["elevation"] for c in cells]; imp=[c["importance"] for c in cells]
        r={"job_id":jid,"file":filename,"point_count":len(pts),"original_point_count":original,"downsampled":downsampled,
           "cell_count":len(cells),"processing_time_s":round(time.perf_counter()-t0,3),"ai_enabled":ai_ok,"model":model,
           "budget":budget,"resolution_counts":rc,"semantic_counts":sem,"motion_enabled":bool(use_previous_frame and LAST is not None),
           "metrics":{"high_cells":rc["0.05"],"medium_cells":rc["0.15"],"low_cells":rc["0.50"],"min_elevation_m":round(float(np.min(elev)),3),"max_elevation_m":round(float(np.max(elev)),3),"avg_elevation_m":round(float(np.mean(elev)),3),"avg_importance":round(float(np.mean(imp)),3)},
           "benchmark":benchmark(pts,cells,grid_t),"cells":cells}
        JOBS[jid].update(r); setstage(jid,"complete",100,"Perception complete.")
        LAST=pts.copy()
    except Exception as e:
        JOBS[jid].update({"stage":"error","progress":100,"message":f"{type(e).__name__}: {e}","error":str(e)})
    finally:
        try: os.remove(path)
        except: pass

@app.post("/api/v3/process")
async def process(file:UploadFile=File(...),budget:str=Form("balanced"),use_previous_frame:bool=Form(False)):
    global JOBS
    ext=os.path.splitext(file.filename)[1].lower()
    if ext not in [".pcd",".ply",".bin",".xyz"]: raise HTTPException(400,"Supported: .pcd .ply .bin .xyz")
    data=await file.read()
    if len(data)>250*1024*1024: raise HTTPException(413,"File is over 250 MB. Use a smaller scan for the laptop demo.")
    fd,path=tempfile.mkstemp(suffix=ext); os.close(fd)
    with open(path,"wb") as f: f.write(data)
    jid=str(uuid.uuid4())
    JOBS[jid]={"job_id":jid,"file":file.filename,"stage":"queued","progress":2,"message":"Queued for AI perception…","created":time.time()}
    threading.Thread(target=worker,args=(jid,path,file.filename,budget,use_previous_frame),daemon=True).start()
    return {"job_id":jid,"stage":"queued","progress":2,"message":"Queued for AI perception…"}

@app.get("/api/v3/jobs/{jid}")
def job(jid:str):
    if jid not in JOBS: raise HTTPException(404,"Job not found")
    return {k:v for k,v in JOBS[jid].items() if k not in ("cells",)}

@app.get("/api/v3/maps/{jid}")
def result(jid:str):
    if jid not in JOBS: raise HTTPException(404,"Not found")
    if JOBS[jid].get("stage")!="complete": raise HTTPException(409,"Job not complete")
    return JOBS[jid]

@app.get("/api/v3/demo")
def demo():
    rng=np.random.default_rng(42)
    road=np.column_stack([rng.uniform(-25,25,14000),rng.uniform(-14,14,14000),rng.normal(0,.025,14000)]).astype(np.float32)
    vehicle=np.column_stack([rng.uniform(3,7,3200),rng.uniform(1,4,3200),rng.uniform(.1,1.6,3200)]).astype(np.float32)
    person=np.column_stack([rng.normal(-2,.35,700),rng.normal(1,.3,700),rng.uniform(.1,1.8,700)]).astype(np.float32)
    veg=np.column_stack([rng.uniform(-13,-8,2300),rng.uniform(3,9,2300),rng.uniform(.2,5.5,2300)]).astype(np.float32)
    pts=np.vstack([road,vehicle,person,veg]); labels=np.zeros(len(pts),dtype=int)
    names=np.array(["road"]*len(pts),dtype=object); groups=np.array(["road"]*len(pts),dtype=object)
    names[14000:17200]="car"; groups[14000:17200]="vehicle"; names[17200:17900]="person"; groups[17200:17900]="pedestrian"; names[17900:]="vegetation"; groups[17900:]="vegetation"
    conf=np.where(groups=="road",.92,np.where(groups=="vehicle",.94,np.where(groups=="pedestrian",.88,.84)))
    cells=assign(aggregate(pts,labels,conf,names.tolist(),groups.tolist()),"balanced",{})
    rc={"0.05":0,"0.15":0,"0.50":0}; sem={}
    for c in cells: rc[f'{c["resolution_m"]:.2f}']+=1; sem[c["semantic_class"]]=sem.get(c["semantic_class"],0)+1
    elev=[c["elevation"] for c in cells]; imp=[c["importance"] for c in cells]
    return {"job_id":"demo","file":"DEMO_SCENE","point_count":len(pts),"original_point_count":len(pts),"downsampled":False,"cell_count":len(cells),"processing_time_s":0.18,"ai_enabled":False,"model":"Synthetic demonstration scene (not an AI benchmark)","budget":"balanced","resolution_counts":rc,"semantic_counts":sem,"motion_enabled":False,"metrics":{"high_cells":rc["0.05"],"medium_cells":rc["0.15"],"low_cells":rc["0.50"],"min_elevation_m":round(float(np.min(elev)),3),"max_elevation_m":round(float(np.max(elev)),3),"avg_elevation_m":round(float(np.mean(elev)),3),"avg_importance":round(float(np.mean(imp)),3)},"benchmark":benchmark(pts,cells,.18),"cells":cells}

app.mount("/assets",StaticFiles(directory=FRONT),name="assets")
