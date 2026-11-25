#!/usr/bin/env bash
# One command, fresh Ubuntu 22.04 GPU machine -> retrained models, evaluations, Gazebo/PX4
# flights and a results.zip.
#
#   git clone https://github.com/arunvaradarajan6/Instruction-Guided-3D-Path-Planning.git igprm-3d
#   # upload datasets.zip to ~ first (scp), then:
#   nohup bash igprm-3d/scripts/cloud_run.sh > ~/run.log 2>&1 &
#   tail -f ~/run.log            # or: cat ~/ig3d_status.txt
#
# Every stage writes a marker when it finishes, so rerunning the same command after a
# crash or reboot continues where it stopped. Delete ~/.ig3d_stages/<stage>.done to redo one.
#
# Knobs (environment variables):
#   DATASETS_ZIP=~/datasets.zip   the datasets/ folder zipped on your Mac
#   PX4_TAG=v1.15.4               PX4-Autopilot release
#   WORKERS=3                     parallel training / evaluation jobs
#   SIM_N=4                       blocks per held-out city for the flights
#   SIM_SOURCES="heldout"         or e.g. "seattle_downtown" to split flights over machines
#   SIM_WIND=1                    also fly the wind batch (0 to skip)
#   SKIP_TRAIN=1                  flights only (models_v2 must already be in datasets.zip)
#   SKIP_SIM=1                    training + evaluation only
set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/src"
DATASETS_ZIP="${DATASETS_ZIP:-$HOME/datasets.zip}"
PX4_TAG="${PX4_TAG:-v1.15.4}"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
WORKERS="${WORKERS:-3}"
SIM_N="${SIM_N:-4}"
SIM_SOURCES="${SIM_SOURCES:-heldout}"
SIM_WIND="${SIM_WIND:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_SIM="${SKIP_SIM:-0}"
ST="$HOME/.ig3d_stages"
LOGS="$REPO/logs"
STATUS="$HOME/ig3d_status.txt"
VENV="$REPO/.venv"
SIM_MODELS="oracle k1_onehot_real k3_onehot_real k3_nli_adapter_real"
SIM_WIND_MODELS="oracle k3_onehot_real k3_nli_adapter_real"

export PX4_DIR IG3D_CORPUS=v2 DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
mkdir -p "$ST" "$LOGS"

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

status() { echo "$(date '+%F %T')  $*" | tee -a "$STATUS"; }

stage() {  # stage NAME critical|optional
  # Each stage runs in its own bash process with `set -e`, so any failing command fails
  # the stage (bash ignores set -e inside functions called from if/&& contexts).
  local name=$1 kind=$2
  if [ -f "$ST/$name.done" ]; then echo "-- skip $name (done)"; return 0; fi
  status "START $name"
  local t0=$SECONDS
  bash "$SELF" --run-stage "$name" >> "$LOGS/$name.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    touch "$ST/$name.done"
    status "OK    $name ($(( (SECONDS - t0) / 60 )) min)"
    return 0
  fi
  status "FAIL  $name -> $LOGS/$name.log"
  tail -n 25 "$LOGS/$name.log" | sed 's/^/      | /'
  if [ "$kind" = critical ]; then status "stopping: fix the error above and rerun"; exit 1; fi
  return 1
}

py() { "$VENV/bin/python" "$@"; }

# ------------------------------------------------------------------------- stages

preflight() {
  . /etc/os-release
  echo "OS: $PRETTY_NAME"
  [ "${VERSION_ID:-}" = "22.04" ] || echo "WARNING: built for Ubuntu 22.04; this is $VERSION_ID"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  local free_gb
  free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  echo "free disk: ${free_gb} GB"
  [ "$free_gb" -ge 40 ] || { echo "need >= 40 GB free disk"; exit 1; }
  [ -f "$DATASETS_ZIP" ] || [ -d "$REPO/datasets/models" ] || {
    echo "missing $DATASETS_ZIP (upload the zipped datasets/ folder first)"; exit 1; }
}

apt_base() {
  if [ -z "$SUDO" ] && ! command -v sudo >/dev/null; then apt-get update && apt-get install -y sudo; fi
  $SUDO apt-get update
  $SUDO apt-get install -y git python3-venv python3-pip unzip zip curl tmux lsb-release
}

unpack_data() {
  if [ -d "$REPO/datasets/models" ]; then echo "datasets/ already present"; return 0; fi
  unzip -q "$DATASETS_ZIP" -d "$REPO"
  [ -d "$REPO/datasets/models" ] || { echo "zip did not contain datasets/models"; exit 1; }
  find "$REPO/datasets" -name .DS_Store -delete
}

install_px4() {
  if [ ! -d "$PX4_DIR" ]; then
    git clone --recursive -b "$PX4_TAG" https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR"
  fi
  # NuttX (flight-controller hardware) toolchain is not needed for simulation
  bash "$PX4_DIR/Tools/setup/ubuntu.sh" --no-nuttx
}

build_px4() {
  cd "$PX4_DIR"
  # build only: `make px4_sitl gz_x500` would also start the simulator and never return
  make px4_sitl
  test -x build/px4_sitl_default/bin/px4
}

python_env() {
  python3 -m venv "$VENV"
  py -m pip install -U pip wheel
  local drv idx
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ "${drv:-0}" -ge 570 ]; then idx=cu128; else idx=cu124; fi
  echo "driver $drv -> torch $idx"
  py -m pip install torch --index-url "https://download.pytorch.org/whl/$idx"
  py -m pip install -r "$SRC/requirements.txt"
  py -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
}

unit_tests() { cd "$SRC" && py -m pytest -m "not slow"; }

language_models() {
  cd "$SRC" && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 py -m ig3d.experiments e3 --out "$REPO/datasets/results_v2"
}

lidar_download() {
  # the AWS Open Data copy of 3DEP; rockyweb.usgs.gov (`download`) throttles to ~50 KB/s
  cd "$SRC" && py -m ig3d.cities fetch-aws --cities all
}

lidar_build() {
  cd "$SRC"
  py -m ig3d.cities build --cities all
  py -m ig3d.cities summary
  for c in seattle_downtown miami_brickell; do
    py -c "from ig3d.cities import blocks_file; import sys; sys.exit(0 if blocks_file('$c').exists() else 1)" \
      || { echo "held-out city $c has no blocks; the flights need it"; exit 1; }
  done
}

sim_smoke() {
  cd "$SRC"
  local d="$REPO/datasets/sim/smoke"
  rm -rf "$d"
  py -m ig3d.sim.run mission --source seattle_downtown --index 0 --model oracle --cls level --out "$d"
  timeout 900 "$VENV/bin/python" -m ig3d.sim.run fly "$d"   # timeout cannot run the py function
  py -m ig3d.sim.run analyse "$d" | tee "$d/analyse.out"
  py - "$d" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1] + "/metrics.json"))
f = json.load(open(sys.argv[1] + "/flight.json"))
print("success", m["success"], "flown_regret", m.get("flown_regret"), "origin", f["origin_source"])
sys.exit(0 if m["success"] else 1)
EOF
}

train_all() { cd "$SRC" && py -m ig3d.multiseed train --workers "$WORKERS"; }

eval_synthetic() {
  cd "$SRC"
  py -m ig3d.multiseed eval --workers "$WORKERS"
  py -m ig3d.multiseed report
}

eval_cities() {
  cd "$SRC"
  for c in austin seattle_downtown miami_brickell; do
    py -m ig3d.lidar ladder --city "$c"
    for s in held hard; do
      py -m ig3d.lidar eval --city "$c" --sentences "$s" --workers "$WORKERS"
      py -m ig3d.lidar report --city "$c" --sentences "$s"
    done
  done
}

sim_dry() {
  cd "$SRC"
  # shellcheck disable=SC2086
  py -m ig3d.sim.run batch --run dry_v2 --sources $SIM_SOURCES --n "$SIM_N" --backend dryrun --models $SIM_MODELS
  py -m ig3d.sim.run report --run dry_v2
}

sim_px4() {
  cd "$SRC"
  # shellcheck disable=SC2086
  py -m ig3d.sim.run batch --run heldout_v2 --sources $SIM_SOURCES --n "$SIM_N" --backend px4 --models $SIM_MODELS
  py -m ig3d.sim.run report --run heldout_v2
}

sim_px4_wind() {
  cd "$SRC"
  # shellcheck disable=SC2086
  py -m ig3d.sim.run batch --run heldout_v2_wind --sources $SIM_SOURCES --n "$SIM_N" --backend px4 \
    --models $SIM_WIND_MODELS --wind 6 --gust 0.4
  py -m ig3d.sim.run report --run heldout_v2_wind
}

package() {
  cd "$REPO"
  rm -f "$HOME/results.zip" "$HOME/models_v2.zip"
  cp "$STATUS" logs/ 2>/dev/null || true
  zip -qr "$HOME/results.zip" datasets/results_v2 datasets/sim logs \
    datasets/cache/lidar/cities/*/blocks_s4.json -x "*.npz" -x "*/px4.log"
  if [ -d datasets/models_v2 ]; then
    zip -qr "$HOME/models_v2.zip" datasets/models_v2 -x "*/resume.pt"
  fi
  ls -lh "$HOME"/*.zip
}

# --------------------------------------------------------------------------- run

if [ "${1:-}" = --run-stage ]; then
  set -e
  "$2"
  exit 0
fi

status "=== ig3d cloud run (repo $REPO) ==="
stage preflight      critical
stage apt_base       critical
stage unpack_data    critical
stage python_env     critical
stage unit_tests     critical
stage language_models critical
stage lidar_download critical
stage lidar_build    critical

SIM_OK=0
if [ "$SKIP_SIM" != 1 ]; then
  stage install_px4 optional \
    && stage build_px4 optional \
    && stage sim_smoke optional \
    && SIM_OK=1
  [ "$SIM_OK" = 1 ] || status "Gazebo/PX4 smoke test did not pass; training continues, flights will be skipped"
fi

if [ "$SKIP_TRAIN" != 1 ]; then
  stage train_all      critical
  stage eval_synthetic optional
  stage eval_cities    optional
fi

if [ "$SIM_OK" = 1 ]; then
  stage sim_dry optional
  stage sim_px4 optional
  [ "$SIM_WIND" = 1 ] && stage sim_px4_wind optional
fi

rm -f "$ST/package.done"
stage package critical
status "=== DONE. Download ~/results.zip (and ~/models_v2.zip), then TERMINATE the instance ==="
