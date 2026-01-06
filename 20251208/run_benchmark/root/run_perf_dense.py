import os
for key, value in os.environ.items():
    print(f"start env {key}={value}")

import sys
import signal
import torch
from threading import Thread
import requests
import time
import pathlib
import logging
import json
import asyncio
import subprocess
import socket
import threading
import concurrent.futures
import random
from transformers import AutoTokenizer, PreTrainedTokenizer

from typing import List, Dict, Optional

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.parent.absolute()))

# from uvicorn.loops.uvloop import uvloop_setup
# uvloop_setup()

tokenizer: Optional[PreTrainedTokenizer] = None
next_random_seed = int(os.environ.get("RANDOM_SEED", 42))
max_seq_len = 10000 # int(os.environ["MAX_SEQ_LEN"], 10000)

def script_exit(pgrp_set: bool = False):
    sys.stdout.flush()
    if pgrp_set:
        os.killpg(0, signal.SIGKILL)
        os._exit(0)
    else:
        os._exit(0)

class ResponseInfo:
    success: bool = False
    input_len: int = 0
    output_len: int = 0

    wait_time: float = 0.0
    total_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    decode_time_per_token: float = 0.0

    """
    output example:
    {
        "response": [
            "aaa"
        ],
        "finished": true,
        "aux_info": [
            {
                "cost_time": 122667.0,
                "iter_count": 500,
                "prefix_len": 0,
                "input_len": 2049,
                "reuse_len": 0,
                "output_len": 500,
                "step_output_len": 500,
                "fallback_tokens": 0,
                "fallback_times": 0,
                "first_token_cost_time": 6129.027,
                "wait_time": 5021.9,
                "pd_sep": false,
                "cum_log_probs": [
                    1.401298464324817e-45
                ],
                "beam_responses": [],
                "softmax_probs": []
            }
        ]
    }
    """

    def __init__(self, response: dict, success: bool = True):
        if not success:
            return
        self.success = success
        aux_info = response.get("aux_info", [])[0]
        self.input_len = aux_info.get("input_len", 0)
        self.output_len = aux_info.get("output_len", 0)
        self.wait_time = aux_info.get("wait_time", 0.0)
        self.total_time = aux_info.get("cost_time", 0.0) - self.wait_time
        self.prefill_time = aux_info.get("first_token_cost_time", 0.0) - self.wait_time
        self.decode_time = self.total_time - self.prefill_time
        self.decode_time_per_token = self.decode_time / (self.output_len - 1)

def test_request(port: int, length: int, output_length: int = 1, enable_torch_profiler=False) -> ResponseInfo:
    global tokenizer
    if not tokenizer:
        raise RuntimeError("tokenizer not initialized!")

    global next_random_seed
    random.seed(next_random_seed)
    #next_random_seed += 1

    prompt_token_ids = [random.randint(100, tokenizer.vocab_size - 100) for _ in range(length)]

    prompt_str = "hello " * (length-1) # tokenizer.decode(prompt_token_ids)
    
    request = {
        "prompt": prompt_str,
        "generate_config": {
            "top_p": 0.1,
            "top_k": 1,
            "temperature": 1.0,
            "max_new_tokens": output_length,
            "min_new_tokens": output_length,
            "num_return_sequences": 1,
            "timeout_ms": 1000000000,
            "gen_timeline": enable_torch_profiler
        }
    }

    try:
        # print(f"start request.")
        response = requests.post(f"http://127.0.0.1:{port}/", json=request)
        if response.status_code != 200:
            print(f"request failed: {response.content}")
            return ResponseInfo({}, False)
        # print(json.dumps(response.json(), indent=4, ensure_ascii=False))
        return ResponseInfo(response.json())
    except Exception as e:
        print(f" request exception:: {e}")
        return ResponseInfo({}, False)

def analyze_results(responses: List[ResponseInfo]):
    total_request_count = len(responses)
    success_requests = [r for r in responses if r.success]
    success_count = len(success_requests)
    fail_count = total_request_count - success_count

    print(f"total requests: {success_count}, succces requests: {success_count}, fail requests: {fail_count}")

    if success_count:
        avg_input_len = sum([r.input_len for r in success_requests]) / len(success_requests)
        avg_output_len = sum([r.output_len for r in success_requests]) / len(success_requests)

        avg_wait_time = sum([r.wait_time for r in success_requests]) / len(success_requests)
        max_wait_time = max([r.wait_time for r in success_requests])
        avg_total_time = sum([r.total_time for r in success_requests]) / len(success_requests)
        max_total_time = max([r.total_time for r in success_requests])
        avg_prefill_time = sum([r.prefill_time for r in success_requests]) / len(success_requests)
        prefill_time_max = max([r.prefill_time for r in success_requests])
        prefill_time_var = sum([(r.prefill_time - avg_prefill_time) ** 2 for r in success_requests]) / len(success_requests)
        avg_decode_time = sum([r.decode_time_per_token for r in success_requests]) / len(success_requests)
        decode_time_max = max([r.decode_time_per_token for r in success_requests])
        decode_time_var = sum([(r.decode_time_per_token - avg_decode_time) ** 2 for r in success_requests]) / len(success_requests)

        print(f"avg input len: {avg_input_len}, avg output len: {avg_output_len}")
        print(f"avg wait time: {avg_wait_time:2f} ms, max wait time: {max_wait_time:2f} ms")
        print(f"avg total time: {avg_total_time:2f} ms, max total time: {max_total_time:2f} ms")
        print(f"avg prefill time: {avg_prefill_time:2f} ms, max prefill time: {prefill_time_max:2f} ms, var: {prefill_time_var:2f}")
        print(f"avg decode time: {avg_decode_time:2f} ms, max decode time: {decode_time_max:2f} ms, var: {decode_time_var:2f}")
    print("============================================================================================================", flush=True)


def _run_concurrency_one_time(test_length, concurrency, output_length, enable_torch_profiler):
    responses = []
    futures = [executor.submit(test_request, port, test_length, output_length, enable_torch_profiler) for _ in range(concurrency)]
    for future in concurrent.futures.as_completed(futures):
        responses.append(future.result())
    return responses

def _run_concurrency_lasting(test_length, concurrency, output_length, num_requests, enable_torch_profiler):
    interval = 1
    sent_count = 0
    futures = []
    responses = []
    while sent_count < num_requests:
        remaining = num_requests - sent_count
        this_concurrency = min(concurrency, remaining)
        futures.extend([executor.submit(test_request, port, test_length, output_length, enable_torch_profiler)
                            for _ in range(this_concurrency)])
        sent_count += this_concurrency

        if sent_count < num_requests:
            time.sleep(interval)
    for future in concurrent.futures.as_completed(futures):
        responses.append(future.result())
    return responses


def run_batch_test(test_lengths: List[int], concurrencies: List[int], output_length: int = 1, num_requests: Optional[int] = None, enable_torch_profiler = False):
    global max_seq_len
    for test_length in test_lengths:
        if test_length > max_seq_len:
            print(f"skip test length: {test_length}, exceed max seq length: {max_seq_len}")
            continue
        for concurrency in concurrencies:
            with torch.cuda.nvtx.range(f"seqlen={test_length},concurrency={concurrency},output={output_length}"):
                print(f"test length: {test_length}, concurrency: {concurrency}, output length: {output_length}")
                if num_requests is not None:
                    responses = _run_concurrency_lasting(test_length, concurrency, output_length, num_requests, enable_torch_profiler)
                else:
                    responses = _run_concurrency_one_time(test_length, concurrency, output_length, enable_torch_profiler)
                analyze_results(responses)
            torch.cuda.nvtx.range_pop()

if __name__ == '__main__':
    port = int(os.environ["START_PORT"])

    pgrp_set = False
    try:
        os.setpgrp()
        pgrp_set = True
    except Exception as e:
        logging.info(f"setpgrp error: {e}")


    tokenizer_path = os.environ.get("TOKENIZER_PATH", os.environ.get("CHECKPOINT_PATH", None))
    if not tokenizer_path:
        raise RuntimeError("no tokenizer path provided!")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"initilize tokenizer [{tokenizer_path}] done.")

    random.seed(next_random_seed)

    #torch.cuda.profiler.start()

    dp_size = int(os.environ.get("DP_SIZE", 1))
    nsight_perf = bool(int(os.environ.get("NSIGHT_PERF", 0)))
    enable_torch_profiler = bool(int(os.environ.get("ENABLE_TORCH_PROFILER", 0)))

    bs = int(os.environ.get("BATCH_SIZE", 0))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1024)

    # test
    test_lengths = [4000]
    output_length = 2000
    concurrencies = [bs] if bs > 0 else [1]
    num_requests = None
    run_batch_test(test_lengths, concurrencies, output_length, num_requests, enable_torch_profiler)

    #torch.cuda.profiler.stop()
    print("request done.")

    #script_exit(pgrp_set)

