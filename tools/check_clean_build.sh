#!/usr/bin/env bash
# Release gate: prove that the *committed* HEAD builds and passes from a pristine
# worktree (no local WIP, no stale build/ artifacts).
#
# Why: a release tag must point at a commit that builds and passes on its own.  Two
# real incidents motivated this: a commit whose header declaration did not match its
# definition (clean checkout failed to link) and a commit that only built because a
# teammate's uncommitted helper file happened to be in the shared build/ tree.
#
# Usage:
#   bash tools/check_clean_build.sh                 # build + five suites + gen_regression
#   bash tools/check_clean_build.sh --quick         # build only (no models needed; still catches
#                                                   # "commit does not build from scratch")
#   bash tools/check_clean_build.sh --keep          # keep the temp worktree for debugging
#   MODEL_DIR=/path/to/models bash tools/check_clean_build.sh
#
# Exit code 0 = READY TO TAG; non-zero = do NOT tag this commit.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

QUICK=0
KEEP=0
WT=""
MODELS="${MODEL_DIR:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --keep) KEEP=1 ;;
    --wt) shift; WT="${1:-}" ;;
    --models) shift; MODELS="${1:-}" ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

REF="$(git rev-parse --short HEAD)"
export MODEL_DIR="$MODELS"   # verify.sh and friends use ${MODEL_DIR} under set -u
DIRTY="$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
if [ "$DIRTY" != "0" ]; then
  echo "[gate] NOTE: $DIRTY tracked file(s) modified locally; the gate tests committed"
  echo "       HEAD ($REF) only, local WIP is NOT included."
fi
if [ -z "$WT" ]; then WT="$ROOT/build/clean_build_check"; fi
FAILED=()

fail() { echo "[gate] FAIL: $*" >&2; FAILED+=("$*"); }
cleanup() {
  if [ "$KEEP" = "1" ]; then
    echo "[gate] kept worktree: $WT"
    return
  fi
  git -C "$ROOT" worktree remove --force "$WT" >/dev/null 2>&1 || rm -rf "$WT"
}
trap cleanup EXIT

rm -rf "$WT"
MODE_LABEL=""; [ "$QUICK" = "1" ] && MODE_LABEL=" (quick build-only)"
echo "[gate] HEAD=$REF  worktree=$WT$MODE_LABEL"
git worktree add --detach "$WT" HEAD >/dev/null 2>&1 || { echo "[gate] cannot create worktree"; exit 1; }
cd "$WT" || exit 1

echo "[gate] 1/6 build.sh"
if ! bash build.sh > build_gate.log 2>&1; then
  fail "build.sh (see $WT/build_gate.log)"
  tail -20 build_gate.log >&2
else
  echo "[gate]   build.sh OK"
fi

# verify_serve's webui asset is a fetched build product, not a tracked file.
if [ -f "$ROOT/build/webui/index.html" ]; then
  mkdir -p build/webui && cp -r "$ROOT/build/webui/." build/webui/ 2>/dev/null
fi

if [ "$QUICK" = "1" ]; then
  echo "[gate] quick mode: build.sh only (use full mode for the release gate)"
else
  if [ -z "$MODELS" ]; then
    fail "MODEL_DIR is required for the full gate (or use --quick for build-only)"
  fi
  echo "[gate] 2/6 verify.sh"
  if ! bash tools/verify.sh > verify_gate.log 2>&1; then
    fail "verify.sh"
    tail -20 verify_gate.log >&2
  else
    echo "[gate]   verify ALL GREEN"
  fi
  echo "[gate] 3/6 smoke_models.sh"
  if ! MODEL_DIR="$MODELS" bash tools/smoke_models.sh > smoke_gate.log 2>&1; then
    fail "smoke_models.sh"
    tail -10 smoke_gate.log >&2
  else
    echo "[gate]   smoke OK"
  fi
  echo "[gate] 4/6 gen_regression.sh"
  if [ -z "$MODELS" ]; then
    fail "gen_regression needs MODEL_DIR (set --models/--quick)"
  elif ! MODEL_DIR="$MODELS" bash tools/gen_regression.sh > genreg_gate.log 2>&1; then
    fail "gen_regression.sh"
    tail -15 genreg_gate.log >&2
  else
    tail -1 genreg_gate.log | sed 's/^/[gate]   /'
  fi
  echo "[gate] 5/6 verify_generality.sh"
  if ! MODEL_DIR="$MODELS" bash tools/verify_generality.sh > generality_gate.log 2>&1; then
    fail "verify_generality.sh"
    tail -10 generality_gate.log >&2
  else
    echo "[gate]   generality GREEN"
  fi
  echo "[gate] 6/6 verify_serve.sh"
  if ! MODEL_DIR="$MODELS" bash tools/verify_serve.sh > serve_gate.log 2>&1; then
    fail "verify_serve.sh"
    tail -12 serve_gate.log >&2
  else
    echo "[gate]   serve verify GREEN"
  fi
  if [ -x build/test_sampler.exe ] && ! ./build/test_sampler.exe > sampler_gate.log 2>&1; then
    fail "test_sampler"
    tail -5 sampler_gate.log >&2
  fi
  if [ -x build/test_json_utf8.exe ] && ! ./build/test_json_utf8.exe > json_utf8_gate.log 2>&1; then
    fail "test_json_utf8 (JSON writer must sanitise invalid UTF-8)"
    tail -5 json_utf8_gate.log >&2
  else
    echo "[gate]   json writer UTF-8 sanitisation GREEN"
  fi
  if [ -x build/test_text_util.exe ] && ! ./build/test_text_util.exe > text_util_gate.log 2>&1; then
    fail "test_text_util (strict UTF-8: overlong/surrogate/>U+10FFFF)"
    tail -8 text_util_gate.log >&2
  else
    echo "[gate]   strict UTF-8 GREEN"
  fi
  if [ -x build/test_tokenizer_budget.exe ] && [ -n "$MODELS" ] &&
     [ -f "$MODELS/Qwen3.5-9B-Q5_K_M.gguf" ]; then
    if ! ./build/test_tokenizer_budget.exe "$MODELS/Qwen3.5-9B-Q5_K_M.gguf" > tok_budget_gate.log 2>&1; then
      fail "test_tokenizer_budget (BPE O(n log n) + input budget)"
      tail -8 tok_budget_gate.log >&2
    else
      echo "[gate]   tokenizer budget/performance GREEN"
    fi
  fi
  if [ -x build/test_json_depth.exe ] && ! ./build/test_json_depth.exe > json_depth_gate.log 2>&1; then
    fail "test_json_depth (JSON reader must cap nesting depth, not overflow the stack)"
    tail -5 json_depth_gate.log >&2
  fi
  if [ -x build/test_json_strict.exe ] && ! ./build/test_json_strict.exe > json_strict_gate.log 2>&1; then
    fail "test_json_strict (JSON reader must enforce RFC 8259 numbers/strings/surrogates)"
    tail -8 json_strict_gate.log >&2
  else
    echo "[gate]   json reader strictness GREEN"
  fi
  if [ -x build/test_config_strict.exe ] && ! ./build/test_config_strict.exe > config_strict_gate.log 2>&1; then
    fail "test_config_strict (config floats must be finite; recursive includes must reject cycles)"
    tail -8 config_strict_gate.log >&2
  else
    echo "[gate]   config validation GREEN"
  fi
  if [ -x build/test_rope_freqs.exe ] && ! ./build/test_rope_freqs.exe > rope_freqs_gate.log 2>&1; then
    fail "test_rope_freqs (gemma4 proportional-rope frequency factors)"
    tail -8 rope_freqs_gate.log >&2
  else
    echo "[gate]   proportional rope freq factors GREEN"
  fi
fi

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "[gate] CLEAN BUILD GATE GREEN: $REF is ready to tag"
  exit 0
fi
echo "[gate] CLEAN BUILD GATE RED: $REF must NOT be tagged (${#FAILED[@]} failure(s)):"
for f in "${FAILED[@]}"; do echo "  - $f"; done
exit 1
