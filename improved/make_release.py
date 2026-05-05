import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "release" / "FakeNewsDetection-Improved"


INCLUDE = [
    "README.md",
    "requirements.txt",
    "app.py",
    "gui.py",
    "main_hybrid.py",
    "prediction_logic.py",
    "decision_engine.py",
    "predict_model.py",
    "news_data.csv",
    "improved",
    "templates",
    "static",
]


def copy_path(src: Path, dst: Path):
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    if RELEASE_ROOT.exists():
        shutil.rmtree(RELEASE_ROOT)
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)

    for rel in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            continue
        dst = RELEASE_ROOT / rel
        copy_path(src, dst)

    # Create empty artifacts folder (user will generate after download)
    (RELEASE_ROOT / "improved_artifacts").mkdir(parents=True, exist_ok=True)

    print("Release folder created at:", RELEASE_ROOT)


if __name__ == "__main__":
    main()

