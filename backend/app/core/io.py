import os, numpy as np, open3d as o3d

def load_points(path):
    ext=os.path.splitext(path)[1].lower()
    if ext==".bin":
        a=np.fromfile(path,dtype=np.float32)
        cols=4 if len(a)%4==0 else 3
        return a.reshape(-1,cols)[:,:3]
    if ext==".xyz":
        return np.loadtxt(path,dtype=np.float32)[:,:3]
    if ext in (".pcd",".ply"):
        pc=o3d.io.read_point_cloud(path)
        return np.asarray(pc.points,dtype=np.float32)
    raise ValueError("Supported formats: .pcd, .ply, .bin, .xyz")
