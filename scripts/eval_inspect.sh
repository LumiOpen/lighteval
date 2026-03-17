#!/bin/bash
# filepath: /home/kai/tw/lib/lighteval_lumiopen/scripts/eval_inspect.sh
#
# Run lighteval evaluations with inspect-ai backend and a scorer model
#
# Usage: bash eval_inspect.sh <model_path> <task_name> [max_tokens]
#
# Example:
#   bash eval_inspect.sh /path/to/model math_500 32768

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash eval_inspect.sh <model_path> <task_name> [max_tokens]"
    echo ""
    echo "Arguments:"
    echo "  model_path  - Path or HuggingFace model name to evaluate"
    echo "  task_name   - Name of the evaluation task (e.g., math_500)"
    echo "  max_tokens  - Maximum tokens for generation (default: 32768)"
    echo ""
    echo "Example:"
    echo "  bash eval_inspect.sh /path/to/model math_500 32768"
    exit 1
fi

MODEL=$1
TASK=$2
MAX_TOKENS=${3:-32768}  # Default to 32k if not specified

# Output directory for results
OUTPUT_DIR=data/inspect_evals/$(basename $MODEL)
mkdir -p $OUTPUT_DIR

# Scorer model configuration
SCORER_MODEL="Qwen/Qwen3.5-9B"
SCORER_PORT=8123
SCORER_BASE_URL="http://localhost:${SCORER_PORT}/v1"

# GPU configuration
SCORER_GPUS="0,1"        # GPUs for scorer model
EVAL_GPUS="2,3,4,5,6,7"  # GPUs for evaluation
SCORER_DP_SIZE=2         # Data parallel size for scorer
EVAL_DP_SIZE=6          # Data parallel size for evaluation

# Generation parameters
TEMPERATURE=0.6
TOP_P=0.95
EPOCHS=1

# vLLM environment variables
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ============================================================================
# Start Scorer Model Server
# ============================================================================

echo "=========================================="
echo "Starting scorer model server"
echo "  Model: $SCORER_MODEL"
echo "  Port: $SCORER_PORT"
echo "  GPUs: $SCORER_GPUS"
echo "  Data Parallel Size: $SCORER_DP_SIZE"
echo "=========================================="

export SCORER_MODEL_PATH=$SCORER_MODEL
export SCORER_MODEL_BASE_URL=$SCORER_BASE_URL

CUDA_VISIBLE_DEVICES=$SCORER_GPUS \
HIP_VISIBLE_DEVICES=$SCORER_GPUS \
vllm serve $SCORER_MODEL \
    --port $SCORER_PORT \
    --data-parallel-size=$SCORER_DP_SIZE \
    --default-chat-template-kwargs '{"enable_thinking": false}' &

VLLM_PID=$!
echo "Scorer server started (PID: $VLLM_PID)"

# Wait for server to initialize
echo "Waiting for scorer server to initialize..."
sleep 120

# Verify server is running
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: Scorer server failed to start or crashed"
    exit 1
fi

# Test server health
if ! curl -s http://localhost:${SCORER_PORT}/health > /dev/null; then
    echo "ERROR: Scorer server health check failed"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "✓ Scorer server is ready"

# ============================================================================
# Run Evaluation
# ============================================================================

echo ""
echo "=========================================="
echo "Running evaluation"
echo "  Model: $MODEL"
echo "  Task: $TASK"
echo "  Max Tokens: $MAX_TOKENS"
echo "  GPUs: $EVAL_GPUS"
echo "  Data Parallel Size: $EVAL_DP_SIZE"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$EVAL_GPUS \
HIP_VISIBLE_DEVICES=$EVAL_GPUS \
lighteval eval vllm/$MODEL $TASK \
    --display plain \
    --log-dir $OUTPUT_DIR \
    --model-args data_parallel_size=$EVAL_DP_SIZE \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --epochs $EPOCHS \
    --epochs-reducer mean \
    --max-tokens $MAX_TOKENS

EVAL_EXIT_CODE=$?

# ============================================================================
# Cleanup
# ============================================================================

echo ""
echo "=========================================="
echo "Shutting down scorer server (PID: $VLLM_PID)"
echo "=========================================="

kill $VLLM_PID 2>/dev/null || echo "Scorer server already terminated"
wait $VLLM_PID 2>/dev/null

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation completed successfully"
else
    echo "✗ Evaluation failed with exit code: $EVAL_EXIT_CODE"
fi
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

exit $EVAL_EXIT_CODE