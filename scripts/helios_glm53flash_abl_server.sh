#!/usr/bin/env bash
# Helios launcher for glm53flash_abl. Uses both GPUs, fp16 KV and the BF16 vision frontend.
# Helios currently has no compressed KV or MTP decoding implementation.
set -euo pipefail

# Works from scripts/ in the checkout and when copied into ~/models/.
SCRIPT_PARENT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$SCRIPT_PARENT/CMakeLists.txt" && -d "$SCRIPT_PARENT/src" ]]; then
    DEFAULT_HELIOS_DIR=$SCRIPT_PARENT
else
    DEFAULT_HELIOS_DIR=$HOME/projects/new_engine/helios-glm53flash
fi
HELIOS_DIR=${HELIOS_DIR:-$DEFAULT_HELIOS_DIR}
BIN=${BIN:-$HELIOS_DIR/build/helios}
MODEL_DIR=${MODEL_DIR:-$HOME/models/glm53flash_abl}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
CAP=${CAP:-262144}
# Reserve more GPU0 memory for image encoding than the 8192-row text benchmark.
CHUNK=${CHUNK:-4096}
PREFIX_SNAP_MB=${PREFIX_SNAP_MB:-4096}
PREFIX_INTERVAL=${PREFIX_INTERVAL:-8192}
API_KEY=${API_KEY:-}
REASONING_EFFORT=${REASONING_EFFORT:-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
HELIOS_VISION_PYTHON=${HELIOS_VISION_PYTHON:-$HOME/shared-venv-gpu/bin/python}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-900}
VRAM_WAIT=${VRAM_WAIT:-60}
MIN_FREE_MIB=${MIN_FREE_MIB:-16000}
VISION_HEADROOM_MIB=${VISION_HEADROOM_MIB:-3072}

usage() {
    cat <<EOF
usage: $0 [start|stop|restart|status] [--host ADDR] [--port N] [--cap N] [--chunk N] [--api-key KEY]

Model: $MODEL_DIR
Engine: $BIN
Defaults: context=$CAP, chunk=$CHUNK, vision=on, KV=fp16, MTP=off.
API: http://$HOST:$PORT/v1 (model id: glm-5.3-flash-exl3)
Vision interpreter: $HELIOS_VISION_PYTHON
Environment overrides: HELIOS_DIR, BIN, MODEL_DIR, HOST, PORT, CAP, CHUNK,
HELIOS_VISION_PYTHON, CUDA_VISIBLE_DEVICES, PREFIX_SNAP_MB, PREFIX_INTERVAL,
REASONING_EFFORT, API_KEY, STARTUP_TIMEOUT, LOG_FILE, PID_FILE, GPU0_NEED_MIB,
MIN_FREE_MIB, VISION_HEADROOM_MIB, VRAM_WAIT.
Images use base64 data:image/... URLs. KV compression and MTP are unavailable in Helios.
EOF
}

COMMAND=start
while (($#)); do
    case "$1" in
        start|stop|restart|status) COMMAND=$1 ;;
        --host|--port|--cap|--chunk|--api-key)
            (($# >= 2)) || { echo "$1 requires a value" >&2; exit 2; }
            case "$1" in
                --host) HOST=$2 ;; --port) PORT=$2 ;; --cap) CAP=$2 ;;
                --chunk) CHUNK=$2 ;; --api-key) API_KEY=$2 ;;
            esac
            shift ;;
        --host=*) HOST=${1#*=} ;; --port=*) PORT=${1#*=} ;;
        --cap=*) CAP=${1#*=} ;; --chunk=*) CHUNK=${1#*=} ;;
        --api-key=*) API_KEY=${1#*=} ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done

for name in PORT CAP CHUNK STARTUP_TIMEOUT VRAM_WAIT MIN_FREE_MIB VISION_HEADROOM_MIB PREFIX_INTERVAL; do
    value=${!name}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
done
[[ "$PREFIX_SNAP_MB" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "PREFIX_SNAP_MB must be a nonnegative integer" >&2; exit 2; }
(( PORT <= 65535 )) || { echo "PORT must be <= 65535" >&2; exit 2; }
(( CAP <= 1048576 )) || { echo "CAP exceeds the model's 1048576-token window" >&2; exit 2; }

# Compute after parsing --cap / --chunk, including the larger quant's trunk and vision headroom.
GPU0_NEED_MIB=${GPU0_NEED_MIB:-$(( 512 + 6850 + CHUNK * 1287 / 1000 + CAP * 11874 / 1000000 + VISION_HEADROOM_MIB ))}
[[ "$GPU0_NEED_MIB" =~ ^[1-9][0-9]*$ ]] || { echo "GPU0_NEED_MIB must be a positive integer" >&2; exit 2; }
PID_FILE=${PID_FILE:-${XDG_RUNTIME_DIR:-/tmp}/helios-glm53flash-abl-${PORT}.pid}
LOG_FILE=${LOG_FILE:-${XDG_RUNTIME_DIR:-/tmp}/helios-glm53flash-abl-${PORT}.log}
SERVER_LOCK_FILE=${SERVER_LOCK_FILE:-${XDG_RUNTIME_DIR:-/tmp}/helios-port-${PORT}.lock}
case "$HOST" in
    0.0.0.0) HEALTH_HOST=127.0.0.1 ;; ::) HEALTH_HOST=::1 ;; *) HEALTH_HOST=$HOST ;;
esac
if [[ "$HEALTH_HOST" == *:* ]]; then
    HEALTH_URL="http://[$HEALTH_HOST]:$PORT/health"
else
    HEALTH_URL="http://$HEALTH_HOST:$PORT/health"
fi

# Stop the engine and any active vision child together.
set -m
proc_start_time() { [[ -r "/proc/$1/stat" ]] && awk '{print $22}' "/proc/$1/stat"; }
is_running() {
    [[ -s "$PID_FILE" ]] || return 1
    local pid expected_start current_start cmdline env_model env_port
    read -r pid expected_start <"$PID_FILE" || return 1
    [[ "$pid" =~ ^[0-9]+$ && "$expected_start" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    current_start=$(proc_start_time "$pid") || return 1
    [[ "$current_start" == "$expected_start" ]] || return 1
    [[ -r "/proc/$pid/cmdline" && -r "/proc/$pid/environ" ]] || return 1
    cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    env_model=$(tr '\0' '\n' <"/proc/$pid/environ" | awk -F= '$1 == "MODEL_DIR" {sub(/^[^=]*=/, ""); print}')
    env_port=$(tr '\0' '\n' <"/proc/$pid/environ" | awk -F= '$1 == "PORT" {sub(/^[^=]*=/, ""); print}')
    [[ "$cmdline" == *"$BIN serve "* && "$env_model" == "$MODEL_DIR" && "$env_port" == "$PORT" ]]
}
read_pid() { local pid start; read -r pid start <"$PID_FILE"; printf '%s\n' "$pid"; }
signal_group() { kill "-$2" -- "-$1" 2>/dev/null || kill "-$2" "$1" 2>/dev/null || true; }
wait_ready() {
    local pid=$1 deadline=$((SECONDS + STARTUP_TIMEOUT))
    while (( SECONDS < deadline )); do
        kill -0 "$pid" 2>/dev/null || return 1
        curl --fail --silent --max-time 2 "$HEALTH_URL" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}
wait_vram() {
    local deadline=$((SECONDS + VRAM_WAIT)) f0 f1
    local -a devices
    IFS=, read -r -a devices <<<"$CUDA_VISIBLE_DEVICES"
    (( ${#devices[@]} == 2 )) || { echo "Helios requires two visible GPUs" >&2; return 1; }
    while :; do
        f0=$(nvidia-smi -i "${devices[0]}" --query-gpu=memory.free --format=csv,noheader,nounits)
        f1=$(nvidia-smi -i "${devices[1]}" --query-gpu=memory.free --format=csv,noheader,nounits)
        if (( f0 >= GPU0_NEED_MIB && f1 >= MIN_FREE_MIB )); then return 0; fi
        if (( SECONDS >= deadline )); then
            echo "insufficient VRAM: GPU0 has $f0 MiB (needs $GPU0_NEED_MIB including vision); GPU1 has $f1 MiB (needs $MIN_FREE_MIB)" >&2
            echo "Free GPU memory or reduce CHUNK/CAP before starting." >&2
            return 1
        fi
        sleep 2
    done
}

mkdir -p "$(dirname "$SERVER_LOCK_FILE")"
exec 9>"$SERVER_LOCK_FILE"
flock -n 9 || { echo "another Helios command is running for port $PORT" >&2; exit 1; }

start() {
    if is_running; then echo "Helios is already running ($(read_pid))"; return 0; fi
    [[ -x "$BIN" ]] || { echo "engine binary not found: $BIN" >&2; return 1; }
    [[ -f "$MODEL_DIR/config.json" ]] || { echo "model config not found: $MODEL_DIR/config.json" >&2; return 1; }
    if ss -H -ltn "sport = :$PORT" | rg -q .; then
        echo "port $PORT is already in use; stop its server first" >&2; return 1
    fi
    "$HELIOS_VISION_PYTHON" -c 'from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel; import torch, safetensors, numpy; from PIL import Image' || {
        echo "vision dependencies missing; see $HELIOS_DIR/requirements-vision.txt" >&2; return 1;
    }
    wait_vram
    mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"
    rm -f "$PID_FILE"
    [[ ! -s "$LOG_FILE" ]] || mv -f "$LOG_FILE" "$LOG_FILE.1"
    local args=(serve "$MODEL_DIR" --host "$HOST" --port "$PORT" --cap "$CAP" --chunk "$CHUNK"
                --prefix-snap-mb "$PREFIX_SNAP_MB" --prefix-interval "$PREFIX_INTERVAL")
    [[ -z "$API_KEY" ]] || args+=(--api-key "$API_KEY")
    [[ -z "$REASONING_EFFORT" ]] || args+=(--reasoning-effort "$REASONING_EFFORT")
    export MODEL_DIR PORT HOST CAP CHUNK HELIOS_VISION_PYTHON CUDA_VISIBLE_DEVICES
    nohup "$BIN" "${args[@]}" >>"$LOG_FILE" 2>&1 9>&- &
    local pid=$! start_time
    start_time=$(proc_start_time "$pid") || { echo "engine exited before recording its PID" >&2; return 1; }
    printf '%s %s\n' "$pid" "$start_time" >"$PID_FILE"
    echo "loading glm53flash_abl: context=$CAP vision=on KV=fp16 MTP=off; log: $LOG_FILE"
    if ! wait_ready "$pid"; then
        echo "server failed readiness check" >&2
        signal_group "$pid" TERM
        sleep 2
        signal_group "$pid" KILL
        rm -f "$PID_FILE"
        tail -n 30 "$LOG_FILE" >&2
        return 1
    fi
    echo "started Helios pid=$pid bound=$HOST:$PORT context=$CAP vision=on KV=fp16 MTP=off"
}
stop() {
    if ! is_running; then rm -f "$PID_FILE"; echo "Helios glm53flash_abl is stopped"; return 0; fi
    local pid; pid=$(read_pid)
    signal_group "$pid" TERM
    for _ in {1..30}; do
        if ! is_running; then rm -f "$PID_FILE"; echo "stopped Helios glm53flash_abl"; return 0; fi
        sleep 1
    done
    signal_group "$pid" KILL
    rm -f "$PID_FILE"
}
status() {
    if ! is_running; then echo "Helios glm53flash_abl is stopped"; return 1; fi
    echo "Helios is running ($(read_pid)) model=$MODEL_DIR bound=$HOST:$PORT"
    curl --fail --silent --max-time 2 "$HEALTH_URL"; echo
}
case "$COMMAND" in
    start) start ;; stop) stop ;; restart) stop; start ;; status) status ;;
esac
