import time, math, psutil, os, numpy as np

def uniform_grid(points,res):
    x=np.floor(points[:,0]/res).astype(np.int64);y=np.floor(points[:,1]/res).astype(np.int64)
    return len(np.unique(np.stack([x,y],axis=1),axis=0))

def benchmark(points,adaptive_cells,adaptive_time):
    rows=[]
    for res in (.05,.15,.50):
        t=time.perf_counter();n=uniform_grid(points,res);dt=time.perf_counter()-t
        rows.append({"method":f"Uniform {int(res*100)} cm","resolution_m":res,"cells":n,
                     "grid_time_s":round(dt,4)})
    # Equivalent adaptive cell count uses area ratio relative to a 5cm reference.
    eq=sum((c["resolution_m"]/.05)**-2 for c in adaptive_cells)
    fixed5=next(x["cells"] for x in rows if x["resolution_m"]==.05)
    reduction=(1-eq/max(fixed5,1))*100
    rows.append({"method":"AI Adaptive","resolution_m":None,"cells":len(adaptive_cells),
                 "equivalent_5cm_cells":round(eq,1),"grid_time_s":round(adaptive_time,4)})
    return {"baselines":rows,"estimated_cell_work_reduction_vs_5cm_pct":round(reduction,2),
            "note":"Cell-work reduction is a structural estimate; timing values are measured locally and are not claimed as GPU speedups."}
