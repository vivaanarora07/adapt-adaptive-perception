import numpy as np, open3d as o3d

def estimate_motion(prev_pts, cur_pts, voxel=.5):
    """ICP ego-registration + nearest-neighbor residual motion.
    Returns cell-level dynamic probabilities. Useful prototype, not object tracking."""
    if prev_pts is None or len(prev_pts)<20 or len(cur_pts)<20:return {},None
    a=o3d.geometry.PointCloud();a.points=o3d.utility.Vector3dVector(prev_pts)
    b=o3d.geometry.PointCloud();b.points=o3d.utility.Vector3dVector(cur_pts)
    ad=a.voxel_down_sample(voxel);bd=b.voxel_down_sample(voxel)
    reg=o3d.pipelines.registration.registration_icp(
        ad,bd,2.0,np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint())
    aligned=np.asarray(ad.transform(reg.transformation).points)
    tree=o3d.geometry.KDTreeFlann(bd)
    buckets={}
    for p in aligned:
        _,idx,d2=tree.search_knn_vector_3d(p,1)
        if not idx:continue
        residual=float(np.sqrt(d2[0]))
        prob=float(np.clip((residual-.20)/1.0,0,1))
        k=(round(float(p[0])),round(float(p[1])))
        buckets.setdefault(k,[]).append(prob)
    return {k:float(np.mean(v)) for k,v in buckets.items()},reg.transformation.tolist()
