## Fake News Detection (Downloadable Package)

This folder is a self-contained “improved” package of your project’s core ML pipeline.

### What’s included
- `improved/prepare_dataset.py`: builds `improved_artifacts/train.csv` + `val.csv` and injects **reliable-source** headlines from `news_data.csv`
- `improved/train_baseline.py`: trains a calibrated baseline model and saves `improved_artifacts/baseline_model.joblib`
- `improved/evaluate.py`: evaluates on validation split
- `improved/predict.py`: CLI prediction
- `predict_model.py`: compatibility wrapper used by older scripts

### What you need to provide
- Put your `dataset.csv` next to this README (same folder).
- Keep `news_data.csv` next to this README (same folder) to provide reliable-source domains/headlines.

### Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run (baseline)

```bash
python improved\prepare_dataset.py
python improved\train_baseline.py
python improved\evaluate.py --model baseline
python improved\predict.py --text "Regional hospital implements telemedicine services for remote patients"
```

### Zip this folder (PowerShell)

```powershell
Compress-Archive -Path ".\*" -DestinationPath "..\FakeNewsDetection-Improved.zip" -Force
```

**Label convention**: 0=real, 1=fake

