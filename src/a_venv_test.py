'''
This is a script to check if the venv is working properly.
'''

import sys
import importlib

def check_dependencies():
    dependencies = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "tqdm",
        "yfinance",
        "plotly",
        "statsmodels"
    ]
    
    print(f"Python Version: {sys.version}")
    print(f"Exec Path: {sys.executable}")
    print("-" * 30)
    
    missing = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"[OK] {dep}")
        except ImportError:
            print(f"[MISSING] {dep}")
            missing.append(dep)
            
    print("-" * 30)
    if not missing:
        print("Venv Health Check: PASSED")
        return True
    else:
        print(f"Venv Health Check: FAILED (Missing: {', '.join(missing)})")
        return False

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)
