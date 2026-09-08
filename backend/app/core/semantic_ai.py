import os, urllib.request, numpy as np

URL="https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth"
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CKPT=os.path.join(ROOT,"models","randlanet_semantickitti_202201071330utc.pth")

# SemanticKITTI learning-map output indices (19 train classes)
TRAIN_NAMES=["car","bicycle","motorcycle","truck","other-vehicle","person","bicyclist","motorcyclist",
            "road","parking","sidewalk","other-ground","building","fence","vegetation","trunk","terrain","pole","traffic-sign"]
GROUP={
"car":"vehicle","bicycle":"two_wheeler","motorcycle":"two_wheeler","truck":"vehicle","other-vehicle":"vehicle",
"person":"pedestrian","bicyclist":"pedestrian","motorcyclist":"pedestrian","road":"road","parking":"road",
"sidewalk":"sidewalk","other-ground":"terrain","building":"building","fence":"structure",
"vegetation":"vegetation","trunk":"vegetation","terrain":"terrain","pole":"infrastructure","traffic-sign":"infrastructure"}
_pipeline=None

def _pipeline_load():
    global _pipeline
    if _pipeline is not None:return _pipeline
    os.makedirs(os.path.dirname(CKPT),exist_ok=True)
    if not os.path.exists(CKPT):
        urllib.request.urlretrieve(URL,CKPT, timeout=20)
    import open3d.ml as _ml3d
    import open3d.ml.torch as ml3d
    # Prefer official config when package data is available.
    pkg=os.path.dirname(_ml3d.__file__)
    cfg_file=os.path.join(pkg,"configs","randlanet_semantickitti.yml")
    cfg=_ml3d.utils.Config.load_from_file(cfg_file)
    model=ml3d.models.RandLANet(**cfg.model)
    pipe=ml3d.pipelines.SemanticSegmentation(model, device="cpu", **cfg.pipeline)
    pipe.load_ckpt(ckpt_path=CKPT)
    _pipeline=pipe
    return pipe

def infer(points):
    try:
        pipe=_pipeline_load()
        pts=np.asarray(points,dtype=np.float32)
        data={"point":pts,"feat":None,"label":np.zeros(len(pts),dtype=np.int32)}
        out=pipe.run_inference(data)
        labels=np.asarray(out["predict_labels"]).reshape(-1).astype(int)
        scores=np.asarray(out["predict_scores"])
        conf=scores.max(axis=1) if scores.ndim==2 else np.asarray(scores).reshape(-1)
        names=[TRAIN_NAMES[x] if 0<=x<len(TRAIN_NAMES) else "other" for x in labels]
        groups=[GROUP.get(n,"other") for n in names]
        return labels, np.asarray(conf,float), names, groups, True, "RandLA-Net / SemanticKITTI"
    except Exception as e:
        # Explicit fallback, surfaced to UI; never represented as AI success.
        # Fast local geometric fallback for custom point clouds when the pretrained
        # model is unavailable or incompatible with the scan. This keeps the
        # adaptive mapping pipeline functional while explicitly reporting that
        # pretrained semantic inference was not used.
        p=np.asarray(points,dtype=np.float32)
        z=p[:,2]
        q10,q35,q70=np.percentile(z,[10,35,70])
        radial=np.sqrt(p[:,0]**2+p[:,1]**2)
        names=[]
        groups=[]
        for zz,rr in zip(z,radial):
            if zz<=q10:
                n,g="road","road"
            elif zz>=q70 and rr<18:
                n,g="structure","structure"
            elif zz>=q35 and rr<12:
                n,g="object","other"
            else:
                n,g="terrain","terrain"
            names.append(n); groups.append(g)
        labels=np.zeros(len(p),dtype=int)
        conf=np.full(len(p),.42,dtype=float)
        return labels,conf,names,groups,False,f"Local geometric fallback: {type(e).__name__}: {e}"
