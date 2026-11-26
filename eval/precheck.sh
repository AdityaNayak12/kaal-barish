#!/bin/sh
# Submission checklist, run before the code freeze from a full checkout that has data/ and the kit installed:
#   sh eval/precheck.sh [port]        (PY=/path/to/python to override the interpreter)
# Clones origin/main fresh WITHOUT Git LFS (as a judge might), then checks: weights are a real file,
# repo_scan has 0 flags, the server starts from the clone, the validator passes, judge_sim passes all 8, repo is public.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
[ -x "$PY" ] || { echo "no python at $PY (set PY=/path/to/python)"; exit 1; }
PORT=${1:-8000}
URL=http://127.0.0.1:$PORT
TMP=$(mktemp -d)
FAILS=0
SERVER=
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null && wait $SERVER 2>/dev/null; rm -rf "$TMP"' EXIT
check() { if [ "$1" = 0 ]; then echo "PASS  $2"; else echo "FAIL  $2"; FAILS=$((FAILS + 1)); fi; }

REMOTE=$(git -C "$REPO" remote get-url origin)
GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.process= -c filter.lfs.smudge= -c filter.lfs.required=false \
  clone -q "$REMOTE" "$TMP/clone"
check $? "fresh clone of $REMOTE at $(git -C "$TMP/clone" log --oneline -1)"

W=$TMP/clone/assets/weights.pt
SIZE=$(($(wc -c < "$W" 2>/dev/null || echo 0)))
# a Git LFS pointer is ~130 bytes of text; real torch weights are a zip starting with "PK"
[ "$SIZE" -gt 500000 ] && [ "$(head -c 2 "$W")" = "PK" ]
check $? "assets/weights.pt is real weights ($SIZE bytes)"

"$PY" -c "
import sys, json; from monsoon_r1.repo_scan import scan
r = scan(sys.argv[1]); print('      scan', r['points'], 'points,', json.dumps(r['flags']))
sys.exit(1 if r['flags'] else 0)" "$TMP/clone"
check $? "repo_scan has 0 flags"

(cd "$TMP/clone" && exec "$PY" serve.py --port "$PORT" > "$TMP/serve.log" 2>&1) &
SERVER=$!
"$PY" -m monsoon_r1.schema_validator "$URL" > "$TMP/validator.log" 2>&1
check $? "schema_validator passes (server running from the fresh clone)"
curl -s -X POST "$URL/health" | grep -q '"kaal-barish-cnn"'
check $? "server loaded the CNN, not the baseline fallback"

"$PY" "$REPO/eval/judge_sim.py" "$URL" > "$TMP/judge.log" 2>&1
check $? "judge_sim: all 8 calls pass"
[ -s "$TMP/serve.log" ] && { echo "      server stderr:"; sed 's/^/        /' "$TMP/serve.log"; }

SLUG=$(echo "$REMOTE" | sed -E 's#.*github\.com[:/]##; s#\.git$##')
[ "$(curl -s -o /dev/null -w '%{http_code}' "https://api.github.com/repos/$SLUG")" = 200 ]
check $? "github.com/$SLUG is public (anonymous API returns 200)"

[ "$FAILS" -gt 0 ] && { echo "--- validator:"; cat "$TMP/validator.log"; echo "--- judge_sim:"; cat "$TMP/judge.log"; }
echo "RESULT: $FAILS failed"
exit "$FAILS"
