import numpy as np

BASE_PRIOR={"pedestrian":1.0,"vehicle":.95,"two_wheeler":.95,"infrastructure":.78,"road":.60,
            "sidewalk":.62,"structure":.42,"building":.32,"terrain":.25,"vegetation":.22,"other":.30}
BUDGETS={
 "quality":{"hi":.58,"mid":.32,"weights":(0.42,0.18,0.16,0.14,0.10)},
 "balanced":{"hi":.72,"mid":.43,"weights":(0.46,0.18,0.14,0.14,0.08)},
 "efficiency":{"hi":.84,"mid":.58,"weights":(0.50,0.16,0.12,0.14,0.08)}
}

def aggregate(points, labels, conf, names, groups, coarse=1.0):
    b={}
    for i,p in enumerate(points):
        k=(int(np.floor(p[0]/coarse)),int(np.floor(p[1]/coarse)))
        b.setdefault(k,[]).append(i)
    cells=[]
    for k,idx in b.items():
        ids=np.asarray(idx); labs=labels[ids]
        vals,cnts=np.unique(labs,return_counts=True); lab=int(vals[np.argmax(cnts)])
        candidates=ids[labs==lab]; rep=int(candidates[np.argmax(conf[candidates])])
        ps=points[ids]; center=ps.mean(axis=0); dist=float(np.linalg.norm(center[:2]))
        cells.append({"x":float(center[0]),"y":float(center[1]),"elevation":float(np.median(ps[:,2])),
          "z_min":float(ps[:,2].min()),"z_max":float(ps[:,2].max()),"height_range":float(np.ptp(ps[:,2])),
          "semantic_label":names[rep],"semantic_class":groups[rep],"confidence":float(np.mean(conf[ids])),
          "point_count":len(idx),"distance_m":dist})
    return cells

def assign(cells,budget="balanced",dynamic_cells=None):
    cfg=BUDGETS.get(budget,BUDGETS["balanced"]); dyn=dynamic_cells or {}
    md=max([c["point_count"] for c in cells] or [1])
    for c in cells:
        sem=BASE_PRIOR.get(c["semantic_class"],.3)
        # Uncertainty is intentionally rewarded: low confidence gets extra precision.
        uncertainty=1.0-c["confidence"]
        proximity=1.0-min(c["distance_m"]/80.0,1.0)
        geometry=min(c["height_range"]/3.0,1.0)
        density=min(c["point_count"]/md,1.0)
        key=(round(c["x"]),round(c["y"]))
        dynamic=float(dyn.get(key,0.0))
        w=cfg["weights"]
        importance=w[0]*sem+w[1]*uncertainty+w[2]*proximity+w[3]*geometry+w[4]*density
        # Measured multi-frame motion has a strong additive contribution.
        importance=min(1.0,importance+0.30*dynamic)
        c.update({"uncertainty":round(uncertainty,4),"dynamic_probability":round(dynamic,4),
                  "importance":round(importance,4)})
        c["resolution_m"]=.05 if importance>=cfg["hi"] else (.15 if importance>=cfg["mid"] else .50)
        if c["resolution_m"]==.05:
            if dynamic>=.5: reason="Dynamic region → fine 5 cm detail to preserve moving-object geometry."
            elif uncertainty>=.55: reason="High semantic uncertainty → fine 5 cm detail for safer interpretation."
            elif sem>=.8 or proximity>=.8: reason="High semantic importance or proximity → fine 5 cm detail."
            else: reason="High combined importance score → fine 5 cm detail."
        elif c["resolution_m"]==.15:
            reason="Moderate importance → 15 cm detail keeps useful structure while reducing compute."
        else:
            reason="Low-priority region → 50 cm detail reduces unnecessary cell-work."
        c["resolution_reason"]=reason
    return cells
