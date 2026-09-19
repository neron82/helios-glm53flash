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
# 512k tokens of KV, the default this server ships with. The engine allocates the cache up front, so
# this is what GPU0 must be able to hold; the cap is limited by GPU0 VRAM, not by the model (whose
# native window is 1M). Measured: KV 3.10 GB at cap 262144 (19.6 GB of GPU0 in total) and 6.20 GB at
# 540000 (22.8 GB), so 524288 leaves ~0.9 GB of headroom. Lower it to free VRAM for other work.
CAP=${CAP:-524288}
# Prefill batch size. Large values only cost scratch memory (~1.8 GB at 8192)
# and make long prompts much faster; short prompts are unaffected because the
# chunk is clamped to the prompt length.
CHUNK=${CHUNK:-8192}
# Empty = no auth. Set for anything reachable beyond localhost.
API_KEY=${API_KEY:-}
# Cross-request prefix caching is on by default and costs no VRAM: the cache planes are
# position-addressed and simply survive between requests, so a prompt that continues or re-sends the
# resident history resumes inside it instead of prefilling from token 0. The KDA recurrence is not
# position-addressed, so its state is snapshotted into *pinned host memory* (142 MB per snapshot, no
# VRAM) every PREFIX_INTERVAL tokens; a request that diverges from the history resumes from the
# newest snapshot at or below the divergence. 4096 MB holds 28 snapshots, ~229k tokens of history at
# the default interval. PREFIX_SNAP_MB=0 keeps extension-only reuse and uses no host memory.
PREFIX_SNAP_MB=${PREFIX_SNAP_MB:-4096}
PREFIX_INTERVAL=${PREFIX_INTERVAL:-8192}
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
# expert pool. Wait up to VRAM_WAIT seconds for both cards to have what they need.
MIN_FREE_MIB=${MIN_FREE_MIB:-16000}
# GPU1 holds the expert pool and sizes it from whatever is free, so a smaller figure still yields a
# working (if slower) server: 16 GB is the floor. GPU0 holds the trunk, the KV cache and the
# per-chunk workspaces, and unlike the pool its need grows with --cap: ~17.1 GB fixed plus ~11.6 KB
# per token of capacity (measured at cap 262144 and at 540000). Waiting for that turns an
# out-of-memory abort into a clear message.
GPU0_NEED_MIB=${GPU0_NEED_MIB:-$(( 17500 + CAP * 11063 / 1000000 ))}
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

# The two cards need different things, so check them separately: GPU0's figure follows --cap
# (GPU0_NEED_MIB) while GPU1 only has a floor (MIN_FREE_MIB).
wait_vram() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local deadline=$((SECONDS + VRAM_WAIT)) f0 f1
    while :; do
        # `tr` leaves no trailing newline, so this read always reports EOF - which `set -e` would
        # treat as a failure and abort the script before a single line of output.
        read -r f0 f1 < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ') || true
        if [[ -n "${f0:-}" && -n "${f1:-}" ]] && (( f0 >= GPU0_NEED_MIB && f1 >= MIN_FREE_MIB )); then
            return 0
        fi
        if (( SECONDS >= deadline )); then
            echo "warning: after ${VRAM_WAIT}s GPU0 has ${f0:-?} MiB free (cap $CAP needs $GPU0_NEED_MIB) and GPU1 has ${f1:-?} MiB (wants $MIN_FREE_MIB) - starting anyway" >&2
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

    local args=(serve "$MODEL_DIR" --host "$HOST" --port "$PORT" --cap "$CAP" --chunk "$CHUNK"
                --prefix-snap-mb "$PREFIX_SNAP_MB" --prefix-interval "$PREFIX_INTERVAL")
    [[ -n "$API_KEY" ]] && args+=(--api-key "$API_KEY")
    [[ -n "$REASONING_EFFORT" ]] && args+=(--reasoning-effort "$REASONING_EFFORT")

    # Export too, so /proc/<pid>/environ carries the identity the checks look for.
    MODEL_DIR="$MODEL_DIR" PORT="$PORT" HOST="$HOST" CAP="$CAP" CHUNK="$CHUNK" \
        REASONING_EFFORT="$REASONING_EFFORT" \
        PREFIX_SNAP_MB="$PREFIX_SNAP_MB" PREFIX_INTERVAL="$PREFIX_INTERVAL" \
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
