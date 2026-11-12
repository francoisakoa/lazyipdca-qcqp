#!/usr/bin/env python3
"""Entry point for the installed native mega campaign."""
from pathlib import Path
import runpy

if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parent/'campaign'/'MEGA.py'),run_name='__main__')
