"""Joint SO(K) rotation math and full toy runner smoke tests."""
from pathlib import Path
import csv,subprocess,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rottoy.rotations import (planes,givens,joint_rotation,joint_path,
    path_weights,signed_permutation_distance,orthogonality_error)
from rottoy.models import plane_rotation

def run_cli(args):
    process=subprocess.run([sys.executable,str(ROOT/"run.py"),*args],
        capture_output=True,text=True,timeout=180,cwd=ROOT)
    if process.returncode:
        raise RuntimeError(f"CLI failed:\n{process.stdout}\n{process.stderr}")
    return process

def test():
    for k in (2,3,4,5):
        pairs=planes(k);d=k*(k-1)//2
        assert len(pairs)==d
        identity=joint_rotation(k,np.zeros(d))
        assert np.allclose(identity,np.eye(k),atol=1e-12)
        for idx,(i,j) in enumerate(pairs):
            angles=np.zeros(d);angles[idx]=23.
            assert np.allclose(joint_rotation(k,angles),plane_rotation(k,i,j,23),atol=1e-12)
        angles=np.linspace(-55,75,d)
        R=joint_rotation(k,angles)
        assert orthogonality_error(R)<1e-12
        assert abs(np.linalg.det(R)-1)<1e-12
        assert signed_permutation_distance(np.eye(k))<1e-12
        for pattern in ("balanced","alternating","ramped","random"):
            assert len(path_weights(k,pattern))==d
            assert np.allclose(joint_path(k,0,pattern),np.eye(k))
            assert orthogonality_error(joint_path(k,33,pattern))<1e-12
    # T8: all four U axes participate, vs one isolated (i,j) Givens.
    R=joint_path(4,25,"ramped")
    assert all(np.count_nonzero(abs(R[row])>1e-3)>=3 for row in range(4))
    assert signed_permutation_distance(R)>.1
    with tempfile.TemporaryDirectory() as td:
        common=["--models","T8-id","--metrics","M018",
                "--u-points","4","--x-points","8",
                "--angles","5","--sweep-steps","7","--output",td]
        run_cli(common+["--rotation-mode","joint-path","--joint-path","ramped"])
        folder=Path(td)/"T8"/"id"/"raw"/"joint_ramped"
        assert (folder/"metric_heatmap.png").exists()
        assert (folder/"metric_curves.png").exists()
        with np.load(folder/"rotation_matrices.npz") as f:
            assert f["rotations"].shape==(5,4,4)
            assert f["angle_pairs"].shape==(6,2)
            assert np.allclose(f["rotations"][2],np.eye(4),atol=1e-12)
            assert not np.allclose(f["rotations"][1],np.eye(4))
        with (folder/"metrics_by_angle.csv").open() as f:
            assert len(list(csv.DictReader(f)))==5
        run_cli(common+["--rotation-mode","joint-search",
                        "--joint-samples","3","--joint-restarts","1",
                        "--joint-maxiter","2"])
        search=Path(td)/"T8"/"id"/"raw"/"joint_search"
        with (search/"joint_search_summary.csv").open() as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==1 and rows[0]["metric"]=="M018"
        assert all(f"phi_{i+1}_{j+1}_deg" in rows[0] for i,j in planes(4))
        assert float(rows[0]["best_found"])<=float(rows[0]["at_identity"])+1e-9
        assert (search/"best_rotation_M018.png").exists()
        assert (search/"joint_search_improvements.png").exists()
        with np.load(search/"best_rotations.npz") as f:
            assert f["rotations"].shape==(1,4,4)
            assert orthogonality_error(f["rotations"][0])<1e-9
    print("PASS: full SO(K) Givens, simultaneous joint path, six-angle search + output plots")

if __name__=="__main__":test()
