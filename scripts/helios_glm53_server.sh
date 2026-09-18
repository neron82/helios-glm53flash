#!/usr/bin/env bash
# Start, stop, restart, or inspect the Helios GLM-5.3-Flash-exl3 server.
#
# Commands: start | stop | restart | status, plus --host / --port / --cap /
# --chunk / --api-key and environment overrides.
# Binds 0.0.0.0 by default so LAN clients can reach it; the readiness probe
# always goes to loopback.
set -euo pipefail

HELIOS_DIR=${HELIOS_DIR:-$HOME/projects/new_engine/helios}
BIN=${BIN:-$HELIOS_DIR/build/helios}

MODEL_DIR=${MODEL_DIR:-$HOME/models/glm53flash}
PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}
# 262144 is the model's native window; the engine allocates its KV cache for
# the full value, so lowering it frees VRAM for more resident experts.
CAP=${CAP:-262144}
# Prefill batch size. Large values only cost scratch memory (~1.8 GB at 8192)
# and make long prompts much faster; short prompts are unaffected because the
# chunk is clamped to the prompt length.
CHUNK=${CHUNK:-8192}
# Empty = no auth. Set for anything reachable beyond localhost.
API_KEY=${API_KEY:-}
# Default reasoning effort for requests that do not set one: low | high | max.
# Measured on a "review this README" prompt with a 4000-token budget:
#   low  -> 488 tokens total,  664 chars of reasoning, 1355 chars of answer,  37 s
#   high -> 1401 tokens total, 3551 chars of reasoning, 1713 chars of answer,  95 s
#   max  -> 4000 tokens burned, 11603 chars of reasoning, NO ANSWER AT ALL, 280 s
# "max" will spend the entire output budget thinking without ever answering. The engine's built-in
# default is already "high"; set this only to override it (low for ordinary chat, max for analysis).
REASONING_EFFORT=${REASONING_EFFORT:-}

SERVER_LOCK_FILE=${SERVER_LOCK_FILE:-${XDG_RUNTIME_DIR:-/tmp}/helios-port-${PORT}.lock}
# Loading 85 GB of weights and pinning the 73 GB arena takes ~40-90 s.
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-300}
# The driver keeps a dead context's VRAM for a few seconds after the process
# exits, so a stop-then-start can otherwise fail outright or silently shrink the
# expert pool. Wait for the smallest free-VRAM figure across the two GPUs to
# reach MIN_FREE_MIB, up to VRAM_WAIT seconds.
MIN_FREE_MIB=${MIN_FREE_MIB:-16000}
VRAM_WAIT=${VRAM_WAIT:-60}

COMMAND=start
while (($#)); do
    case "$1" in
        start|stop|restart|status) COMMAND=$1 ;;
        --host)
            (($# >= 2)) || { echo "--host requires an address" >&2; exit 2; }
            HOST=$2; shift ;;
        --host=*) HOST=${1#*=} ;;
        --port)
            (($# >= 2)) || { echo "--port requires a number" >&2; exit 2; }
            PORT=$2; shift ;;
        --port=*) PORT=${1#*=} ;;
        --cap)
            (($# >= 2)) || { echo "--cap requires a number" >&2; exit 2; }
            CAP=$2; shift ;;
        --cap=*) CAP=${1#*=} ;;
        --chunk)
            (($# >= 2)) || { echo "--chunk requires a number" >&2; exit 2; }
            CHUNK=$2; shift ;;
        --chunk=*) CHUNK=${1#*=} ;;
        --api-key)
            (($# >= 2)) || { echo "--api-key requires a value" >&2; exit 2; }
            API_KEY=$2; shift ;;
        --api-key=*) API_KEY=${1#*=} ;;
        -h|--help)
            echo "usage: $0 [start|stop|restart|status] [--host ADDR] [--port N] [--cap N] [--chunk N] [--api-key KEY]"
            exit 0 ;;
        *)
            echo "usage: $0 [start|stop|restart|status] [--host ADDR] [--port N] [--cap N] [--chunk N] [--api-key KEY]" >&2
            exit 2 ;;
    esac
    shift
done

# Derived from the resolved port, so --port moves them with it. An explicit
# PID_FILE/LOG_FILE in the environment still wins.
PID_FILE=${PID_FILE:-${XDG_RUNTIME_DIR:-/tmp}/helios-glm53-${PORT}.pid}
LOG_FILE=${LOG_FILE:-${XDG_RUNTIME_DIR:-/tmp}/helios-glm53-${PORT}.log}

case "$HOST" in
    0.0.0.0) HEALTH_HOST=127.0.0.1 ;;
    ::) HEALTH_HOST=::1 ;;
    *) HEALTH_HOST=$HOST ;;
esac
if [[ "$HEALTH_HOST" == *:* ]]; then
    HEALTH_URL="http://[${HEALTH_HOST}]:${PORT}/health"
else
    HEALTH_URL="http://${HEALTH_HOST}:${PORT}/health"
fi

proc_start_time() {
    [[ -r "/proc/$1/stat" ]] || return 1
    awk '{print $22}' "/proc/$1/stat"
}

read_pid() {
    local pid start
    read -r pid start <"$PID_FILE"
    printf '%s\n' "$pid"
}

# A PID alone is not proof: it may have been recycled by an unrelated process.
# Re-verify the start time, the command line and the port before believing it.
is_running() {
    [[ -s "$PID_FILE" ]] || return 1
    local pid expected_start current_start cmdline
    read -r pid expected_start <"$PID_FILE" || return 1
    [[ "$pid" =~ ^[0-9]+$ && "$expected_start" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    current_start=$(proc_start_time "$pid") || return 1
    [[ "$current_start" == "$expected_start" ]] || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    [[ "$cmdline" == *"$BIN"* && "$cmdline" == *"serve"* && "$cmdline" == *"$MODEL_DIR"* ]]
}

pid_start_line() {
    local pid=$1 start
    start=$(proc_start_time "$pid") || {
        echo "server process exited before its PID could be recorded" >&2
        return 1
    }
    printf '%s %s\n' "$pid" "$start"
}

# The engine needs ~21.6 GB per card when the pool is large, but it sizes the
# expert pool from whatever is free, so a somewhat smaller figure still yields a
# working (if slower) server. 16 GB is the floor that guarantees it starts.
wait_vram() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local deadline=$((SECONDS + VRAM_WAIT)) free
    while :; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
        if [[ -n "$free" && "$free" -ge "$MIN_FREE_MIB" ]]; then
            return 0
        fi
        if (( SECONDS >= deadline )); then
            echo "warning: only ${free:-?} MiB free per GPU after ${VRAM_WAIT}s (wanted $MIN_FREE_MIB) - starting anyway" >&2
            return 0
        fi
        sleep 2
    done
}

wait_ready() {
    local pid=$1
    for _ in $(seq 1 "$STARTUP_TIMEOUT"); do
        kill -0 "$pid" 2>/dev/null || return 1
        if curl --fail --silent --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

if command -v flock >/dev/null 2>&1; then
    exec 9>"$SERVER_LOCK_FILE"
    if ! flock -n 9; then
        echo "another helios server command is already running" >&2
        exit 1
    fi
fi

start() {
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required for the server readiness probe" >&2
        return 1
    fi
    if [[ ! -x "$BIN" ]]; then
        echo "engine binary not found or not executable: $BIN" >&2
        echo "build it with: cmake -B $HELIOS_DIR/build -G Ninja -S $HELIOS_DIR && cmake --build $HELIOS_DIR/build" >&2
        return 1
    fi
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "model directory not found: $MODEL_DIR" >&2
        return 1
    fi
    if is_running; then
        echo "helios server is already running ($(read_pid))"
        return 0
    fi
    # Never start a second server on the same port - most likely the other
    # model's server is already up.
    if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :$PORT" 2>/dev/null | grep -q .; then
        echo "port $PORT is already in use - stop the other server first" >&2
        return 1
    fi
    wait_vram
    rm -f "$PID_FILE"
    mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
    # Each run gets a clean log; the previous one is kept as .1. Without this, a traceback from an
    # earlier run sits in the file and looks live to anyone tailing it.
    if [[ -s "$LOG_FILE" ]]; then
        mv -f "$LOG_FILE" "$LOG_FILE.1"
    fi

    local args=(serve "$MODEL_DIR" --host "$HOST" --port "$PORT" --cap "$CAP" --chunk "$CHUNK")
    [[ -n "$API_KEY" ]] && args+=(--api-key "$API_KEY")
    [[ -n "$REASONING_EFFORT" ]] && args+=(--reasoning-effort "$REASONING_EFFORT")

    # Export too, so /proc/<pid>/environ carries the identity the checks look for.
    MODEL_DIR="$MODEL_DIR" PORT="$PORT" HOST="$HOST" CAP="$CAP" CHUNK="$CHUNK" \
        REASONING_EFFORT="$REASONING_EFFORT" \
        nohup "$BIN" "${args[@]}" >>"$LOG_FILE" 2>&1 9>&- &
    local pid=$!
    pid_start_line "$pid" >"$PID_FILE"
    if ! wait_ready "$pid"; then
        echo "server failed readiness check; inspect $LOG_FILE" >&2
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -KILL "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
        return 1
    fi
    echo "started helios server pid=$(read_pid) port=$PORT context=$CAP bound=$HOST"
    echo "log: $LOG_FILE"
}

stop() {
    if ! is_running; then
        # The wrapper PID can die while the engine keeps the GPUs: kill any
        # stray helios serve for THIS port before declaring it stopped.
        local stray
        stray=$(pgrep -f "$BIN serve.*--port $PORT" || true)
        if [[ -n "$stray" ]]; then
            echo "reaping stray helios serve process(es): $stray" >&2
            kill $stray 2>/dev/null || true
            sleep 2
            stray=$(pgrep -f "$BIN serve.*--port $PORT" || true)
            [[ -n "$stray" ]] && kill -KILL $stray 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        echo "helios server is not running"
        return 0
    fi
    local pid
    pid=$(read_pid)
    kill "$pid"
    for _ in {1..30}; do
        if ! is_running; then
            rm -f "$PID_FILE"
            echo "stopped helios server"
            return 0
        fi
        sleep 1
    done
    if is_running; then
        echo "server did not stop gracefully; sending SIGKILL" >&2
        kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
}

status() {
    if is_running; then
        echo "helios server is running ($(read_pid)) model=$MODEL_DIR bound=${HOST:-0.0.0.0}:$PORT cap=$CAP"
        if command -v curl >/dev/null 2>&1; then
            curl --fail --silent "$HEALTH_URL" || true
            echo
        fi
    else
        echo "helios server is stopped"
        return 1
    fi
}

case "$COMMAND" in
    start) start ;;
    stop) stop ;;
    restart) stop || true; start ;;
    status) status ;;
esac
