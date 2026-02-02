
import os
from pathlib import Path

def check_file(path, description):
    p = Path(path)
    if p.exists():
        print(f"✅ FOUND: {description} ({p})")
        return True
    else:
        print(f"❌ MISSING: {description} ({p})")
        return False

def verify_setup():
    print("Verifying HIVD Classifier Setup...")
    print("-" * 50)
    
    # Project Root
    root = Path("c:/Users/vasil/Desktop/PROJECTS/A Framework for Efficient Image Categorisation in Social Research/HIVD_classifier")
    
    # Check Backend
    backend = root / "backend"
    check_file(backend / "main.py", "Main API")
    check_file(backend / "classifier_engine.py", "Classifier Engine")
    check_file(backend / "safety_net.py", "Safety Net")
    
    # Check Frontend
    frontend = root / "frontend"
    check_file(frontend / "index.html", "HTML Index")
    check_file(frontend / "style.css", "Modern CSS Styles")
    check_file(frontend / "app.js", "Frontend Application Logic")
    
    # Check Configuration
    check_file(root / "config.yaml", "Configuration")
    
    print("-" * 50)
    print("Verifying Dependencies (Imports)...")
    
    dependencies = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("torch", "PyTorch"),
        ("open_clip", "OpenCLIP"),
        ("ultralytics", "YOLO (Ultralytics)"),
        ("albumentations", "Albumentations"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("aiosqlite", "aiosqlite")
    ]
    
    all_deps_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ INSTALLED: {name}")
        except ImportError as e:
            print(f"❌ MISSING: {name} ({e})")
            all_deps_ok = False
            
    if not all_deps_ok:
        print("\n⚠️  Some dependencies are missing. Please run: pip install -r requirements.txt")
    else:
        print("\n✅ All dependencies are installed!")

    print("-" * 50)
    print("To run the application:")
    print("1. Open terminal")
    print(f"2. cd \"{backend}\"")
    print("3. python main.py")
    print("4. Open http://127.0.0.1:8000 in your browser")

if __name__ == "__main__":
    verify_setup()
