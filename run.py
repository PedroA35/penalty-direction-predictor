"""
CLI Task Runner (Cross-platform alternative to Make).
Usage:
    python run.py help
    python run.py install
    python run.py pose
    python run.py features
    python run.py train
    python run.py test
    python run.py app
    python run.py clean
    python run.py all
"""

import pathlib
import shutil
import subprocess
import sys


COMMANDS = {
    "install": ("Install project dependencies from requirements.txt", [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]),
    "pose": ("Extract pose landmarks from video clips (YOLO-Pose)", [sys.executable, "src/pose.py"]),
    "features": ("Compute biomechanical features and trajectory smoothing", [sys.executable, "src/features.py"]),
    "train": ("Train models and evaluate cross-validation benchmark", [sys.executable, "src/train.py"]),
    "test": ("Run automated pytest unit test suite", [sys.executable, "-m", "pytest"]),
    "app": ("Launch interactive Streamlit web dashboard", ["streamlit", "run", "app/app.py"]),
}


def clean_artifacts():
    """Removes __pycache__ and .pytest_cache directories."""
    count = 0
    for p in pathlib.Path(".").rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
        count += 1
    for p in pathlib.Path(".").rglob(".pytest_cache"):
        shutil.rmtree(p, ignore_errors=True)
        count += 1
    print(f"Cleaned {count} cache artifact directories.")


def print_help():
    print("Available commands:")
    for cmd, (desc, _) in COMMANDS.items():
        print(f"  python run.py {cmd:<10} - {desc}")
    print("  python run.py clean      - Remove Python cache artifacts (__pycache__, .pytest_cache)")
    print("  python run.py all        - Run complete pipeline (features -> train -> test)")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
        print_help()
        return

    target = sys.argv[1].lower()

    if target == "clean":
        clean_artifacts()
        return

    if target == "all":
        for step in ["features", "train", "test"]:
            print(f"\n>>> Running: {step}")
            ret = subprocess.run(COMMANDS[step][1])
            if ret.returncode != 0:
                print(f"[ERROR] Step '{step}' failed with code {ret.returncode}")
                sys.exit(ret.returncode)
        print("\n[SUCCESS] Full pipeline completed successfully.")
        return

    if target not in COMMANDS:
        print(f"[ERROR] Unknown command: '{target}'")
        print_help()
        sys.exit(1)

    cmd = COMMANDS[target][1]
    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
