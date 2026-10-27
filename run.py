"""Standalone entry point; no editable package install required."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from rottoy.__main__ import cli
if __name__=="__main__":cli()
