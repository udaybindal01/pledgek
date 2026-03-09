# =============================================================================
# PLEDGE-KARMA: Full Evaluation Pipeline
# Run this from the pledge_karma/ root directory:
#     cd pledge_karma && bash ../run_all.sh
#
# What this does, in order:
#   Phase 0  — Sanity checks (fail fast if something is missing)
#   Phase 1  — Fit BKT parameters from real ASSISTments data
#   Phase 2  — Run full longitudinal evaluation (real corpus, real encoder)
#   Phase 3  — Validate MRL divergence signal
#   Phase 4  — Validate metacognitive gap signal
#   Phase 5  — Print paper-ready summary table
#
# Expected runtime on Mac M-series: ~25-40 minutes total
#   Phase 1:  ~3 min  (EM fitting on 93k interactions)
#   Phase 2:  ~15 min (100 students × 10 weeks × real embeddings)
#   Phase 3:  ~8 min  (5000 interactions × embedding)
#   Phase 4:  ~2 min  (KARMA simulation on 200 students)
#   Phase 5:  <1 min
# =============================================================================

set -euo pipefail   # Exit on any error

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="outputs/run_all_${TIMESTAMP}.log"
mkdir -p outputs scripts

# Tee all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " PLEDGE-KARMA Full Evaluation Pipeline"
echo " Started: $(date)"
echo " Log: $LOG_FILE"
echo "============================================================"

# =============================================================================
# PHASE 0 — Sanity checks
# =============================================================================
echo ""
log_info "Phase 0: Sanity checks"

# Python
if ! python3 -c "import sys; assert sys.version_info >= (3,9)" 2>/dev/null; then
    log_error "Python 3.9+ required"; exit 1
fi
log_ok "Python $(python3 --version)"

# sentence-transformers
if ! python3 -c "import sentence_transformers" 2>/dev/null; then
    log_error "sentence-transformers not installed."
    log_error "Run: pip install sentence-transformers"
    exit 1
fi
ST_VERSION=$(python3 -c "import sentence_transformers; print(sentence_transformers.__version__)")
log_ok "sentence-transformers $ST_VERSION"

# nomic encoder reachable (just import, don't download yet)
if ! python3 -c "from models.mrl_encoder import MRLEncoder" 2>/dev/null; then
    log_error "Cannot import MRLEncoder. Are you running from pledge_karma/ root?"
    exit 1
fi
log_ok "MRLEncoder importable"

# ASSISTments data
ASSISTMENTS_PATH="${ASSISTMENTS_PATH:-data/raw/assistments.csv}"
if [[ ! -f "$ASSISTMENTS_PATH" ]]; then
    # Try alternative known paths
    for p in "data/raw/interactions.csv" "data/raw/assistments2009.csv"; do
        [[ -f "$p" ]] && ASSISTMENTS_PATH="$p" && break
    done
fi
if [[ ! -f "$ASSISTMENTS_PATH" ]]; then
    log_error "ASSISTments data not found. Tried: $ASSISTMENTS_PATH"
    log_error "Set ASSISTMENTS_PATH=/path/to/your/file or place at data/raw/assistments.csv"
    exit 1
fi
N_LINES=$(wc -l < "$ASSISTMENTS_PATH")
log_ok "ASSISTments data: $ASSISTMENTS_PATH ($N_LINES lines)"

# OpenStax corpus
CONCEPTS_PATH="${CONCEPTS_PATH:-data/processed/math/concepts.json}"
CHUNKS_PATH="${CHUNKS_PATH:-data/processed/math/chunks.json}"
if [[ ! -f "$CONCEPTS_PATH" ]]; then
    for p in "data/processed/concepts.json" "data/processed/physics_v1/concepts.json"; do
        [[ -f "$p" ]] && CONCEPTS_PATH="$p" && break
    done
fi
if [[ ! -f "$CONCEPTS_PATH" ]]; then
    log_error "OpenStax concepts not found. Tried: $CONCEPTS_PATH"
    log_error "Run the OpenStax pipeline first: python data/pipelines/openstax_pipeline.py --book all"
    exit 1
fi
N_CONCEPTS=$(python3 -c "import json; d=json.load(open('$CONCEPTS_PATH')); print(len(d))" 2>/dev/null || echo "?")
log_ok "OpenStax concepts: $CONCEPTS_PATH ($N_CONCEPTS concepts)"

log_ok "All sanity checks passed"

# =============================================================================
# PHASE 1 — Fit BKT parameters from real ASSISTments data
# =============================================================================
echo ""
log_info "Phase 1: Fitting BKT parameters from real ASSISTments data"
log_info "This replaces hardcoded defaults (0.1, 0.15, 0.1, 0.2) with data-fitted values"

BKT_PARAMS_FILE="config/bkt_params.json"

if [[ -f "$BKT_PARAMS_FILE" ]]; then
    log_warn "BKT params already fitted: $BKT_PARAMS_FILE"
    log_warn "Delete this file and re-run to refit. Skipping."
else
    python3 scripts/fit_bkt_params.py \
        --assistments "$ASSISTMENTS_PATH" \
        --output "$BKT_PARAMS_FILE"

    if [[ ! -f "$BKT_PARAMS_FILE" ]]; then
        log_error "BKT fitting failed — $BKT_PARAMS_FILE not created"; exit 1
    fi

    # Auto-update base_config.yaml with fitted params
    python3 - <<'PYEOF'
import json, re
from pathlib import Path

params_file = "config/bkt_params.json"
config_file = "config/base_config.yaml"

with open(params_file) as f:
    params = json.load(f)["global_params"]

print(f"\nFitted BKT params: {params}")

config_text = Path(config_file).read_text()

# Replace each param line
for key, val in params.items():
    pattern = rf"(\s+{key}:\s*)[\d.]+"
    replacement = rf"\g<1>{val}"
    config_text, n = re.subn(pattern, replacement, config_text)
    if n == 0:
        print(f"  WARNING: Could not find {key} in config")
    else:
        print(f"  Updated {key} = {val}")

Path(config_file).write_text(config_text)
print(f"\nUpdated: {config_file}")
PYEOF

    log_ok "BKT parameters fitted and written to config/base_config.yaml"
fi

# =============================================================================
# PHASE 2 — Full longitudinal evaluation (real corpus, real encoder)
# =============================================================================
echo ""
log_info "Phase 2: Full longitudinal evaluation"
log_info "100 students × 10 weeks × real OpenStax corpus × nomic-embed-text-v1.5"
log_warn "Expected runtime: ~15 minutes"

EVAL_OUTPUT="outputs/eval_full_${TIMESTAMP}.json"

python3 experiments/run_experiment.py \
    --mode full \
    --data-source openstax \
    --config config/base_config.yaml

# Find the latest eval output (run_experiment.py names it by its own timestamp)
LATEST_EVAL=$(ls -t outputs/eval_full_*.json 2>/dev/null | head -1)
if [[ -z "$LATEST_EVAL" ]]; then
    # Fallback: find any recent eval output
    LATEST_EVAL=$(ls -t outputs/eval_*.json 2>/dev/null | head -1)
fi

if [[ -z "$LATEST_EVAL" ]]; then
    log_error "No eval output found after running experiment"; exit 1
fi
log_ok "Eval results: $LATEST_EVAL"

# =============================================================================
# PHASE 3 — Validate MRL divergence signal
# =============================================================================
echo ""
log_info "Phase 3: Validating MRL divergence as metacognitive signal"
log_info "Tests: high divergence queries → lower next-question accuracy"
log_warn "Expected runtime: ~8 minutes (5000 interactions × embedding)"

MRL_OUTPUT_DIR="outputs/mrl_validation_${TIMESTAMP}"

# Find math chunks for the corpus argument
MATH_CHUNKS=""
for p in "data/processed/math/chunks.json" "data/processed/chunks.json" \
         "data/processed/physics_v1/chunks.json"; do
    [[ -f "$p" ]] && MATH_CHUNKS="$p" && break
done

if [[ -z "$MATH_CHUNKS" ]]; then
    log_warn "Math chunks not found — MRL validation will use skill names as proxy corpus"
    MATH_CHUNKS="none"
fi

python3 scripts/validate_mrl_divergence.py \
    --assistments "$ASSISTMENTS_PATH" \
    --corpus "$MATH_CHUNKS" \
    --output "$MRL_OUTPUT_DIR" \
    --max-samples 5000

MRL_RESULT=$(python3 -c "
import json
from pathlib import Path
f = Path('$MRL_OUTPUT_DIR/mrl_validation_results.json')
if f.exists():
    d = json.load(open(f))
    print(f\"r={d['point_biserial_correlation']}, p={d['p_value']}, gap={d['accuracy_gap']:.3f}\")
else:
    print('result file not found')
" 2>/dev/null || echo "could not read result")

log_ok "MRL validation complete: $MRL_RESULT"
log_ok "Results: $MRL_OUTPUT_DIR/"

# =============================================================================
# PHASE 4 — Validate metacognitive gap signal
# =============================================================================
echo ""
log_info "Phase 4: Validating metacognitive gap via Dunning-Kruger prediction"
log_info "Tests: overconfident students fail more than BKT alone predicts"
log_warn "Expected runtime: ~2 minutes"

MCG_OUTPUT_DIR="outputs/metacognitive_validation_${TIMESTAMP}"

python3 scripts/validate_metacognitive_gap.py \
    --assistments "$ASSISTMENTS_PATH" \
    --output "$MCG_OUTPUT_DIR" \
    --max-students 200

MCG_RESULT=$(python3 -c "
import json
from pathlib import Path
f = Path('$MCG_OUTPUT_DIR/metacognitive_gap_validation.json')
if f.exists():
    d = json.load(open(f))
    validated = d.get('gap_adds_predictive_value', False)
    oc = d.get('overconfident_bkt_error', 0)
    wc = d.get('wellcalibrated_bkt_error', 0)
    print(f\"validated={validated}, OC_error={oc:.4f}, WC_error={wc:.4f}\")
else:
    print('result file not found')
" 2>/dev/null || echo "could not read result")

log_ok "Metacognitive gap validation: $MCG_RESULT"
log_ok "Results: $MCG_OUTPUT_DIR/"

# =============================================================================
# PHASE 5 — Paper-ready summary table
# =============================================================================
echo ""
log_info "Phase 5: Paper-ready summary"
echo ""

python3 - <<PYEOF
import json
from pathlib import Path

# ── Table 1: Longitudinal eval ───────────────────────────────────────────────
eval_file = sorted(Path("outputs").glob("eval_*.json"), key=lambda p: p.stat().st_mtime)[-1]
with open(eval_file) as f:
    eval_data = json.load(f)

print("=" * 70)
print("TABLE 1: Longitudinal Simulation Results")
print(f"(Source: {eval_file.name})")
print("=" * 70)
print(f"{'Method':<22} {'Admiss':>7} {'NDCG@10':>8} {'MRR':>7} {'MCE':>7} {'Sim.LG':>8}")
print("-" * 70)
for method, r in eval_data.items():
    if isinstance(r, dict) and "admissibility_rate" in r:
        print(f"{method:<22} "
              f"{r['admissibility_rate']:>7.3f} "
              f"{r['ndcg@10']:>8.3f} "
              f"{r['mrr']:>7.3f} "
              f"{r.get('mce', 0):>7.3f} "
              f"{r['sim_learning_gain']:>8.3f}")
print()

# ── Table 2: MRL validation ───────────────────────────────────────────────────
mrl_files = sorted(Path("outputs").glob("mrl_validation_*/mrl_validation_results.json"),
                   key=lambda p: p.stat().st_mtime)
if mrl_files:
    with open(mrl_files[-1]) as f:
        mrl = json.load(f)
    print("=" * 70)
    print("TABLE 2: MRL Divergence Validation")
    print("=" * 70)
    print(f"  Low  divergence → next accuracy:  {mrl['low_divergence_accuracy']:.3f}")
    print(f"  High divergence → next accuracy:  {mrl['high_divergence_accuracy']:.3f}")
    print(f"  Accuracy gap:                     {mrl['accuracy_gap']:+.3f}")
    print(f"  Correlation r:                    {mrl['point_biserial_correlation']:.3f}")
    print(f"  p-value:                          {mrl['p_value']:.4f}")
    validated = "✓ VALIDATED" if mrl["significant"] else "✗ NOT SIGNIFICANT"
    print(f"  Result: {validated}")
    print()

# ── Table 3: Metacognitive gap validation ────────────────────────────────────
mcg_files = sorted(Path("outputs").glob("metacognitive_validation_*/metacognitive_gap_validation.json"),
                   key=lambda p: p.stat().st_mtime)
if mcg_files:
    with open(mcg_files[-1]) as f:
        mcg = json.load(f)
    print("=" * 70)
    print("TABLE 3: Metacognitive Gap Validation")
    print("=" * 70)
    print(f"  BKT prediction error — underconfident: {mcg['underconfident_bkt_error']:.4f}")
    print(f"  BKT prediction error — well-calibrated: {mcg['wellcalibrated_bkt_error']:.4f}")
    print(f"  BKT prediction error — overconfident:   {mcg['overconfident_bkt_error']:.4f}")
    validated = "✓ VALIDATED" if mcg["gap_adds_predictive_value"] else "✗ NOT VALIDATED"
    print(f"  Result: {validated}")
    print()

print("=" * 70)
print(f"Full log: outputs/run_all_{Path('$LOG_FILE').name.split('_',2)[-1]}")
PYEOF

echo ""
echo "============================================================"
echo " PLEDGE-KARMA Evaluation Complete"
echo " Finished: $(date)"
echo " Log: $LOG_FILE"
echo "============================================================"