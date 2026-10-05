#!/usr/bin/env python3
"""
Check if all required dependencies are installed.
"""

import sys


def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = [
        'ultralytics',
        'torch',
        'cv2',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'PIL',
        'yaml',
        'onnx',
        'onnxruntime'
    ]

    web_packages = [
        'streamlit',
        'sqlalchemy',
        'alembic'
    ]

    missing = []
    installed = []
    web_missing = []
    web_installed = []

    print("Checking core Python packages...")
    for package in required_packages:
        try:
            __import__(package)
            installed.append(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} - NOT INSTALLED")

    print("\nChecking web interface packages...")
    for package in web_packages:
        try:
            __import__(package)
            web_installed.append(package)
            print(f"  ✓ {package}")
        except ImportError:
            web_missing.append(package)
            print(f"  ✗ {package} - NOT INSTALLED")

    return missing, installed, web_missing, web_installed


def main():
    print("="*60)
    print("nuts_vision - Dependency Check")
    print("="*60 + "\n")

    missing_packages, installed_packages, web_missing, web_installed = check_python_packages()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if missing_packages:
        print("\n❌ Missing core Python packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nTo install missing packages, run:")
        print("   pip install -r requirements.txt")
    else:
        print("\n✓ All core Python packages installed!")

    if web_missing:
        print("\n⚠️  Missing web interface packages:")
        for pkg in web_missing:
            print(f"   - {pkg}")
        print("\nTo install web interface packages, run:")
        print("   pip install streamlit sqlalchemy alembic")
    else:
        print("\n✓ All web interface packages installed!")

    if missing_packages:
        print("\n⚠ Some core dependencies are missing. Please install them before using nuts_vision.")
        return 1
    elif web_missing:
        print("\n⚠ Core dependencies satisfied, but web interface packages are missing.")
        print("Install them to use the web interface: pip install streamlit sqlalchemy alembic")
        return 0
    else:
        print("\n✓ All dependencies satisfied! You're ready to use nuts_vision.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
