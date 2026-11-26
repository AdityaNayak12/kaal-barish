# Team coordination — DevForge Track 2 "Multiverse of Monsoon" (repo: kaal-barish)
3 Claude Code instances, one repo. Read CONTRACT.md before writing any code.

## Who owns what — only edit your own area
- sarthak: dashboard/, brief/ — map dashboard, shock builder, impact module, 2-page brief
- nayak:   serve.py, engine/, eval/ — API server, fallback model, local scoring harness, repo checks
- pawar:   model/, train/, assets/ — CNN, training, weights, climatology
Shared files (CONTRACT.md, CLAUDE.md, README.md, requirements.txt, .gitignore, .gitattributes): announce in #devforge before editing.

## cross-claude protocol
- Session start: `register` with your name above, then `check_messages` on channel `devforge`.
- Check messages before each new task and after each commit.
- Any change to a shape, file path, function signature or endpoint:
  1. update CONTRACT.md, 2. `message` #devforge, 3. wait for a reply from the affected owner, 4. then change code.
- Need something from another area? Send a `request` to its owner; never edit their files.
- Large content (>500 chars): `share_data`, then message the key.
- Finished a task: short `status` message. New local score: post the points table from eval/score_local.py.

## Track 2 hard rules (breaking these costs points or disqualifies)
- serve.py and everything it imports (engine/, model/) must contain:
  - NO date literals (e.g. 2021-07-15), NO datetime(20xx...) calls
  - NO internet, weather APIs, or the strings mosdac / cdsapi / imdlib / open-meteo
  - NO plotting or UI libraries (plotly, matplotlib, streamlit)
  - NO training code (.backward(), optimizer.step, .fit()) — that lives in train/ only
- Never commit .nc / .grib / .zarr files or raw data. data/ is gitignored.
- No saved array shaped [>400, ..., 32, 36] in the repo (counts as a weather archive). A 366-day climatology is fine.
- Weights go in assets/ via Git LFS (*.pt). Repo must stay under 2 GB.
- Inference must be deterministic: the same input gives exactly the same output (null shock test).
- Output must be finite, rain >= 0, t2m within 233.15–333.15 K.
- The dashboard only talks to the server over HTTP; it never imports server code.
- Never put credentials (MOSDAC, GitHub tokens) in code, prompts or commits.

## Before every merge to main
- `python serve.py --port 8000` + `python -m monsoon_r1.schema_validator http://localhost:8000` must pass.
- Repo scan: `python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"` — no critical flags.

## Git
- Branches: `sarthak/...`, `nayak/...`, `pawar/...`.
- Pull main and merge into main every 30–60 min, small merges, run the server before pushing.
- Never force-push main. Always pull before push.
- About 10–15 commits per person in total. Plain, descriptive messages. No Co-Authored-By or Claude lines.

## Hackathon rules
- All code written during the event; no pre-built projects.
- Working prototype > many features. Dashboard must run on live /simulate output, never pre-rendered results.
- Code freeze: 27 Nov 2026, 09:00 IST (confirm with organisers) — everything merged, weights pushed and repo public before then.
