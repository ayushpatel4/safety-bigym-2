import sys
import os

from pathlib import Path

# Add robobase path to sys.path since it's local
sys.path.insert(0, str(Path("/home/ap2322/Documents/robobase")))

from robobase.workspace import Workspace

print("workspace.py patched and compiled successfully.")
