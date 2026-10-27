"""Smoke test the entire unified CLI and the neural backend on a tiny grid."""
from pathlib import Path
import subprocess,tempfile,sys,csv,json
ROOT=Path(__file__).resolve().parents[1]

def call(args):
    return subprocess.run([sys.executable,str(ROOT/"run.py"),*args],cwd=str(ROOT),
                          capture_output=True,text=True,check=True,timeout=180)

def test():
    a=call(["--list-models"])
    assert "T9-sig" in a.stdout
    b=call(["--list-metrics"])
    assert "M018" in b.stdout
    c=call(["--describe-metric","M018"])
    assert "Formula:" in c.stdout and "Numerical:" in c.stdout
    with tempfile.TemporaryDirectory() as tmp:
        args=["--models","T3-id","T9-sig","--metrics","M001","M006","M017","M018",
              "--pairs","first","--u-points","4","--x-points","8",
              "--angles","5","--sweep-steps","9","--output",tmp]
        call(args)
        for toy,variant in [("T3","id"),("T9","sig")]:
            folder=Path(tmp)/toy/variant/"raw"/"u1_u2"
            csvpath=folder/"metrics_by_angle.csv"
            assert csvpath.exists(),csvpath
            with csvpath.open() as handle:
                data=list(csv.DictReader(handle))
                assert len(data)==5
                assert list(data[0])==["angle_deg","M001","M006","M017","M018"]
            assert (folder/"metric_heatmap.png").exists()
            assert (folder/"metric_curves.png").exists()
        call(["--models","T1-sig","--metrics","M004","M018","--backend","both",
              "--train-neural","--nn-train-samples","64","--nn-val-samples","32",
              "--nn-epochs","3","--nn-width","8","--nn-depth","2",
              "--u-points","3","--x-points","6","--angles","3",
              "--sweep-steps","7","--output",tmp])
        folder=Path(tmp)/"T1"/"sig"/"raw"/"u1_u2"
        assert (folder/"paired_metrics_by_angle.csv").exists()
        assert (folder/"paired_overlay_curves.png").exists()
        assert (Path(tmp)/"checkpoints"/"T1"/"sig"/"mlp.pt").exists()
        cfg=json.loads((folder/"settings.json").read_text())
        assert cfg["variant"]=="T1-sig" and cfg["options"]["backend"]=="both"
    print("PASS: CLI, plots, paired eta/F, 24-item selection, checkpoints")

if __name__=="__main__":test()
