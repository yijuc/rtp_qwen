export TOKENIZER_PATH=/mnt/md0/models/Qwen3-8B
export CHECKPOINT_PATH=/mnt/md0/models/Qwen3-8B
export START_PORT=8008
export NSIGHT_PERF=0
#export HIPBLASLT_LOG_FILE=./Qwen3-8B-FP8-Dynamic-kernel.log
#export HIPBLASLT_LOG_MASK=32

BATCH_SIZE=1 ENABLE_TORCH_PROFILER=1 /opt/conda310/bin/python3 run_perf_dense.py
