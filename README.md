# Clusterer

A lightweight, configurable toolkit for unsupervised clustering of short or long texts. 
It builds vector embeddings for your data, discovers dense regions with HDBSCAN, 
optionally reduces dimensionality for visualization (e.g., t‑SNE), 
and exposes graph-friendly outputs for downstream exploration.

## Highlights

- Embedding-driven clustering of text
- Density-based clustering with HDBSCAN (robust to noise/outliers)
- Optional dimensionality reduction (e.g., t‑SNE) for plotting and inspection
- Graph-oriented outputs for cluster structure exploration
- Configuration-first workflow via JSON files in `configs/`

## Project structure

- `clusterer/` — Python package with the clustering pipeline and utilities
- `configs/` — Ready-to-use JSON configuration examples (e.g., small vs. full)
- `requirements.txt` — Python dependencies
- `install-local.sh` — Convenience script for local setup (optionally to use)
- `setup.py` — Package metadata and local installation helper
- `CHANGELOG.md` — Version history
- `LICENSE` — License information

## Requirements

- Python 3.10+
- A virtual environment (virtualenv recommended)

## Installation

Using virtualenv (recommended):
```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) Install the package locally for import in your apps
pip install -e .
```

Alternatively, you can use the helper script:
```bash
bash install-local.sh
```
Note: Ensure the script matches your environment and adjust if necessary.

## License

This project is distributed under the terms of the license in `LICENSE`.

## Changelog

See `CHANGELOG.md` for the release history.
