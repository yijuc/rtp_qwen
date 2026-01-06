export TOKENIZER_PATH=/data/models/Qwen3-8B-FP8-Dynamic
export CHECKPOINT_PATH=/data/models/Qwen3-8B-FP8-Dynamic
export START_PORT=8001
export NSIGHT_PERF=0

BATCH_SIZE=1 ENABLE_TORCH_PROFILER=0 /opt/conda310/bin/python3 run_perf_dense.py
