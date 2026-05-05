## Fake News Detection (Improved)

This project includes:
- A **baseline** text classifier (TF‑IDF + calibrated linear model)
- A **DistilBERT** trainer/predictor (optional, slower but often more accurate)
- A **reliability-aware dataset builder** that can inject **reliable-source headlines** from `news_data.csv`
- A **hybrid verifier** that can use Fact Check + RSS + (optional) ML fallback

### Quickstart (recommended baseline)

1) Create a virtual environment and install deps:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Build a training dataset (adds reliable-source examples from `news_data.csv`):

```bash
python improved\prepare_dataset.py
```

3) Train baseline model:

```bash
python improved\train_baseline.py
```

4) Evaluate baseline model:

```bash
python improved\evaluate.py --model baseline
```

5) Run a prediction:

```bash
python improved\predict.py --text "Regional hospital implements telemedicine services for remote patients"
```

### DistilBERT (optional)

```bash
python improved\train_distilbert.py
python improved\evaluate.py --model distilbert
```

### Packaging a downloadable zip

```powershell
python improved\make_release.py
Compress-Archive -Path ".\release\FakeNewsDetection-Improved\*" -DestinationPath ".\FakeNewsDetection-Improved.zip" -Force
```

### Label convention (important)

This repo uses: **0 = real, 1 = fake** (everywhere in the improved pipeline).

