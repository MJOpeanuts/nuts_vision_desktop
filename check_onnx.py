#!/usr/bin/env python3
"""Startup check: ONNX must be importable before the app starts (exit 1 otherwise)."""
import sys

missing = []
for name in ("onnx", "onnxruntime"):
    try:
        __import__(name)
    except ImportError:
        missing.append(name)

if missing:
    print(f"ERROR: missing dependencies: {', '.join(missing)}")
    print("Install them with the project's virtual environment:")
    print(f'  "{sys.executable}" -m pip install -r requirements.txt')
    sys.exit(1)
print("ONNX dependencies OK")
