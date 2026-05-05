import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "improved_artifacts"


def _clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_label_to_int(v) -> int | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in {"0", "real", "r", "true", "verified", "legit", "true_news"}:
        return 0
    if s in {"1", "fake", "f", "false", "hoax", "fraud", "misleading"}:
        return 1
    if "fake" in s or "hoax" in s or "false" in s or "mislead" in s:
        return 1
    if "real" in s or "true" in s or "verified" in s or "legit" in s:
        return 0
    return None


def domain_from_url(u: str) -> str:
    try:
        if not u:
            return ""
        p = urlparse(str(u))
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


@dataclass(frozen=True)
class ReliableSource:
    domain: str
    score: float


def load_reliable_sources(news_data_csv: Path) -> dict[str, ReliableSource]:
    if not news_data_csv.exists():
        return {}
    df = pd.read_csv(news_data_csv)
    if "link" not in df.columns:
        return {}
    domains = {}
    for u in df["link"].astype(str).tolist():
        d = domain_from_url(u)
        if not d:
            continue
        domains[d] = ReliableSource(domain=d, score=1.0)
    return domains


def build_from_dataset_csv(dataset_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    title = df["title"] if "title" in df.columns else pd.Series([""] * len(df))
    body = df["text"] if "text" in df.columns else pd.Series([""] * len(df))
    text = (title.fillna("").astype(str) + " " + body.fillna("").astype(str)).map(_clean_text)

    label_raw = None
    for c in ("label", "class", "tags"):
        if c in df.columns:
            label_raw = df[c]
            break
    if label_raw is None:
        label_raw = pd.Series([None] * len(df))

    y = label_raw.map(normalize_label_to_int)
    out = pd.DataFrame({"text": text, "label": y})
    out = out.dropna(subset=["text", "label"])
    out["label"] = out["label"].astype(int)
    out = out[out["text"].str.len() >= 20]
    out = out.drop_duplicates(subset=["text"])
    return out


def build_reliable_real_from_news_data(news_data_csv: Path, reliable_domains: dict[str, ReliableSource]) -> pd.DataFrame:
    if not news_data_csv.exists():
        return pd.DataFrame(columns=["text", "label", "source_domain", "is_reliable_domain", "reliability_score"])
    df = pd.read_csv(news_data_csv)
    if "title" not in df.columns or "link" not in df.columns:
        return pd.DataFrame(columns=["text", "label", "source_domain", "is_reliable_domain", "reliability_score"])
    titles = df["title"].astype(str).map(_clean_text)
    links = df["link"].astype(str)
    domains = links.map(domain_from_url)
    rel = domains.map(lambda d: float(reliable_domains.get(d, ReliableSource(d, 0.0)).score) if d else 0.0)
    out = pd.DataFrame(
        {
            "text": titles,
            "label": 0,
            "source_domain": domains,
            "is_reliable_domain": (rel >= 0.8).astype(int),
            "reliability_score": rel,
        }
    )
    out = out[out["text"].str.len() >= 15]
    out = out.drop_duplicates(subset=["text"])
    return out


def add_reliability_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source_domain"] = ""
    df["reliability_score"] = 0.0
    df["is_reliable_domain"] = 0
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", default=str(ROOT / "dataset.csv"))
    ap.add_argument("--news_data_csv", default=str(ROOT / "news_data.csv"))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reliable_real_multiplier", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = Path(args.dataset_csv)
    news_data_csv = Path(args.news_data_csv)

    reliable_domains = load_reliable_sources(news_data_csv)
    base = add_reliability_features(build_from_dataset_csv(dataset_csv))

    reliable_real = build_reliable_real_from_news_data(news_data_csv, reliable_domains)
    if len(reliable_real) and args.reliable_real_multiplier > 1:
        reliable_real = pd.concat([reliable_real] * int(args.reliable_real_multiplier), ignore_index=True)

    full = pd.concat([base, reliable_real], ignore_index=True)
    full = full.dropna(subset=["text", "label"])
    full["label"] = full["label"].astype(int)
    full = full.drop_duplicates(subset=["text"])

    groups = full["text"].map(lambda s: re.sub(r"\W+", " ", str(s).lower()).strip())
    uniq = pd.DataFrame({"g": groups, "label": full["label"]}).drop_duplicates("g")
    g_train, g_val = train_test_split(
        uniq["g"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=uniq["label"],
    )
    train = full[groups.isin(set(g_train))].reset_index(drop=True)
    val = full[groups.isin(set(g_val))].reset_index(drop=True)

    train.to_csv(out_dir / "train.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    val.to_csv(out_dir / "val.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    full.to_csv(out_dir / "full.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    print("Saved improved_artifacts/{train,val,full}.csv")


if __name__ == "__main__":
    main()

