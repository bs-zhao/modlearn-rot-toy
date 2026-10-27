# modlearn-rot-toy

Independent analytic toy benchmark for functional post-hoc rotations. Native generator parameters and stimulus dimensions lie in [0,1]. Each model supports identity, sigmoid, and square response variants. Reproducible angle sweeps produce per-metric curves, normalized heatmaps, and XU effect maps. The complete implementation is being added under `src/rottoy/`.

Run: `python run.py --models T1 T2 T3 T4 --wrappers all` after `python -m pip install -r requirements.txt`.

Outputs are saved under ignored `outputs/` and never synchronized to GitHub.
