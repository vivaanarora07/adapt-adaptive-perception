ADAPT Adaptive Perception WINNING PROTOTYPE v2

Run: cd backend; python -m pip install -r requirements.txt; uvicorn app.main:app --reload
Open http://127.0.0.1:8000/

Features: RandLA-Net semantic path; semantic map; adaptive 5/15/50cm; elevation graph; circular allocation; importance; uncertainty; optional motion; semantic composition; fixed-resolution benchmark; compute policies; Adaptive Perception demo scene.

Scientific honesty: demo scene is synthetic and not an AI benchmark. Pretrained RandLA-Net is SemanticKITTI-domain. Motion is ICP + residual motion prototype. Cell-work reduction is a structural estimate, not hardware speedup. Final Adaptive Perception claims should use labeled target-domain validation and measured latency/memory.

Team: Code Alpha
