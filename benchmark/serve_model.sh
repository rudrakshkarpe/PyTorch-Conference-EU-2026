#!/usr/bin/env bash
set -euo pipefail

# ── Defaults ──
MODEL=""
PORT=8000
PREFIX_CACHING=false
GPU_MEM_UTIL=0.90
DTYPE="bfloat16"

usage() {
    cat <<EOF
Usage: bash serve_model.sh --model <hf-model-id> [OPTIONS]

Starts a vLLM OpenAI-compatible server for the given model.
Kills any existing vLLM server on the same port first.

Options:
  --model <id>            HuggingFace model ID (required)
  --port <int>            Server port (default: 8000)
  --prefix-caching        Enable automatic prefix caching
  --gpu-mem <float>       GPU memory utilization 0-1 (default: 0.90)
  --dtype <str>           Model dtype (default: bfloat16)
  -h, --help              Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --prefix-caching) PREFIX_CACHING=true; shift ;;
        --gpu-mem)        GPU_MEM_UTIL="$2"; shift 2 ;;
        --dtype)          DTYPE="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *)                echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required"
    usage
fi

# ── Kill existing vLLM server on this port ──
EXISTING_PID=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Killing existing process on port $PORT (PID $EXISTING_PID) ..."
    kill -9 $EXISTING_PID 2>/dev/null || true
    sleep 2
fi

# ── Build vLLM command ──
CMD="vllm serve $MODEL"
CMD+=" --host 0.0.0.0"
CMD+=" --port $PORT"
CMD+=" --dtype $DTYPE"
CMD+=" --gpu-memory-utilization $GPU_MEM_UTIL"
CMD+=" --max-model-len 32768"
CMD+=" --trust-remote-code"

if [ "$PREFIX_CACHING" = true ]; then
    CMD+=" --enable-prefix-caching"
    echo "Prefix caching: ENABLED"
else
    echo "Prefix caching: DISABLED"
fi

echo "Starting vLLM server ..."
echo "  Model:  $MODEL"
echo "  Port:   $PORT"
echo "  Command: $CMD"
echo ""

# ── Launch in background, log to file ──
LOG_FILE="results/vllm_server.log"
mkdir -p results
$CMD > "$LOG_FILE" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
echo "$VLLM_PID" > results/.vllm_pid

# ── Wait for health check ──
echo "Waiting for server to be ready ..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "Server ready on port $PORT (took ${WAITED}s)"
        exit 0
    fi
    sleep 3
    WAITED=$((WAITED + 3))
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Check $LOG_FILE"
        tail -30 "$LOG_FILE"
        exit 1
    fi
done

echo "ERROR: Server did not become ready within ${MAX_WAIT}s"
echo "Last log lines:"
tail -30 "$LOG_FILE"
kill $VLLM_PID 2>/dev/null || true
exit 1
