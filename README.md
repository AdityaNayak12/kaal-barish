# Kaal Barish: Multiverse of Monsoon

Climate digital twin of India for the Yugaantar DevForge "Multiverse of Monsoon" track.
`serve.py` currently holds the kit's placeholder model (repeats the last day). The real model is not built yet.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Download `india_climate_train.nc` and `stats.json` from the
[kit release](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0) into `data/`
(gitignored: the raw data is never committed). Install the kit wheel from the same release for the validator.

## Run

```bash
.venv/bin/python serve.py --port 8000
.venv/bin/python -m monsoon_r1.schema_validator http://localhost:8000
```
