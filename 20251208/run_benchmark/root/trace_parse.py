import json
import sys
import statistics
import os
import re
from pathlib import Path
GPU_TRACE_CAT = ["kernel", "gpu_memcpy", "gpu_memset"]
EMBEDDING_PREFIX = "void rtp_llm::embedding_lookup_kernel"
LAYER_START_PREFIX = "void rtp_llm::addBiasResidual"

PRE_GEMM_MATCHES = ["dynamic_per_token_scaled_quant","Memset","computeFP8Quantize128Kernel"]
GEMM_MATCHES = ["nvjet_", "deep_gemm", "Cijk_Ailk", "_gemm_"] 
POST_GEMM_MATCHES = ["Cijk_SB_BiasS", "cublasLt::splitKreduce_kernel"]

GENERAL_NORM_MATCHES = ["generalRmsNorm", "Rmsnorm2dFwd"]
ATTENTION_MATCHES = ["FmhaFwdKernel", "aiter::pa_", "flash_attention",
                     "BatchPrefillWithPagedKVCacheKernel", 
                     "PersistentVariableLengthMergeStatesKernel",
                     "BatchDecodeWithPagedKVCacheKernel",
                     "paged_attention"]
SILU_MATCHES = ["silu_kernel", "Silu"]
ALL_REDUCE_END_MATCHES = ["cross_device_reduce", "AllReduce"]

NVIDIA_MATCHES = ["deep_gemm", "nvjet_"]

def is_embedding(name):
    return name.startswith(EMBEDDING_PREFIX)


def is_layer_start(name):
    return name.startswith(LAYER_START_PREFIX)


def get_gpu_trace(trace):
    gpu_trace = list()
    for event in trace:
        try:
            if event["cat"] in GPU_TRACE_CAT:
                gpu_trace.append(event)
        except:
            pass
    return gpu_trace


def get_one_forward_trace(gpu_trace, start_index = 0):
    forward_trace = list()
    started = False
    next_start_index = None
    for index, event in enumerate(gpu_trace[start_index:]):
        if not started and is_embedding(event['name']):
            started = True
            continue
        if started:
            if not is_embedding(event['name']):
                forward_trace.append(event)
                continue
            else:
                started = False
                next_start_index = index
                break
    return forward_trace, next_start_index


def get_layer_traces(forward_trace):
    one_layer_trace = []
    layer_traces = []
    started = False
    for event in forward_trace:
        if not started and is_layer_start(event["name"]):
            started = True
            one_layer_trace.append([event["name"], event["dur"], event["ts"]])
            continue
        if started:
            if not is_layer_start(event["name"]):
                one_layer_trace.append([event["name"], event["dur"], event["ts"]])
            else:
                kernels_dur = sum([t[1] for t in one_layer_trace])
                trace_dur = one_layer_trace[-1][2] + one_layer_trace[-1][1] - one_layer_trace[0][2]
                layer_traces.append((trace_dur, kernels_dur, one_layer_trace))
                one_layer_trace = []
                one_layer_trace.append([event["name"], event["dur"], event["ts"]])
    return layer_traces


def match(name, match_list):
    if not isinstance(name, str):
        name = name[0]
    if isinstance(match_list, str):
        match_list = [match_list]
    for m in match_list:
        if m in name:
            return True
    return False

def get_match_indexes(trace, match_list):
    if isinstance(match_list, str):
        match_list = [match_list]
    indexes = list()
    
    for index, (n, t, ts) in enumerate(trace):
        for m in match_list:
            if m in n:
                indexes.append(index)
                break
    return indexes

def __tune_trace(trace):
    new_trace = list()
    # merge elementwise around flashinfer kernel
    for index, (n, t, ts) in enumerate(trace):
        if "void flashinfer::" in n and "elementwise" in trace[index+1][0]:
            t = t + trace[index+1][1]
        if "elementwise" in n and "void flashinfer::" in trace[index-1][0]:
            continue
        new_trace.append((n, t, ts))
    return new_trace


def padding(obj, target):
    for _ in range(target - len(obj)):
        obj.append(("NA", 0, 0))


def get_details(layer_trace):
    trace_dur, kernels_dur, trace = layer_trace
    trace = __tune_trace(trace)
    bubble_dur = trace_dur - kernels_dur

    # configs
    rms_norm_num = 3
    qkv_proj_num = 2
    qk_norm_num = 1
    silu_num = 1
    after_silu_gemm_num=3
    rms_norm = list()
    qk_norm = list()
    qkv_proj = list()
    rotary_emb = list()
    rotary_emb_num = 4
    mha = list()
    mha_num = 2
    o_proj = list()
    o_proj_num = 2
    mha_all_reduce = list()
    mha_all_reduce_num = 2
    post_norm = list()
    post_norm_num = 1
    before_silu_gemm = list()
    before_silu_gemm_num = 6
    silu = list()
    after_silu_gemm = list()
    gemm_all_reduce = list()
    gemm_all_reduce_num = 2
    bubbles = [("bubbles", bubble_dur, 0)]


    # rms_norm
    rms_norm.extend(trace[:3])

    # qkv_proj and qk_norm
    qk_norm_indexes = get_match_indexes(trace, "fusedQkRmsNorm")
    has_qk_norm = len(qk_norm_indexes)>0
    if has_qk_norm:
        qk_norm_index = qk_norm_indexes[0]
    if has_qk_norm:
        qk_norm.append(trace[qk_norm_index])
        qkv_proj.extend(trace[3:qk_norm_index])
        qkv_proj_end_index = qk_norm_index-1
    else:
        qkv_proj_indexes = get_match_indexes(trace[:5], PRE_GEMM_MATCHES + GEMM_MATCHES + POST_GEMM_MATCHES)
        qkv_proj_end_index = qkv_proj_indexes[-1]
        for i in qkv_proj_indexes:
            qkv_proj.append(trace[i])
        
    # mha
    mha_index = get_match_indexes(trace, ATTENTION_MATCHES)[0]
    mha.append(trace[mha_index])
    if match(trace[mha_index+1], ATTENTION_MATCHES):
        mha_end_index = mha_index+1
        mha.append(trace[mha_index+1])
    else:
        mha_end_index = mha_index

    # rotary_emb
    if has_qk_norm:
        rotary_emb.extend(trace[qk_norm_index+1: mha_index])
    else:
        rotary_emb.extend(trace[qkv_proj_end_index+1: mha_index])

    
    # post_norm
    post_norm_index = get_match_indexes(trace, GENERAL_NORM_MATCHES)[-1]
    post_norm.append(trace[post_norm_index])


    # o_proj, mha_all_reduce
    all_reduce_end_indexes = get_match_indexes(trace, ALL_REDUCE_END_MATCHES)
    has_all_reduce = len(all_reduce_end_indexes)>0
    if not has_all_reduce:
        o_proj.extend(trace[mha_end_index+1: post_norm_index])
    else:
        mha_all_reduce_end_index = all_reduce_end_indexes[0]
        if match(trace[mha_all_reduce_end_index-1],"Memcpy"):
            mha_all_reduce_index = mha_all_reduce_end_index-1
        else:
            mha_all_reduce_index = mha_all_reduce_end_index
        mha_all_reduce.extend(trace[mha_all_reduce_index: mha_all_reduce_end_index+1])
        o_proj.extend(trace[mha_end_index+1: mha_all_reduce_index])


    # silu
    silu_index = get_match_indexes(trace, SILU_MATCHES)[0]
    silu.append(trace[silu_index])

    # before_silu_gemm
    before_silu_gemm.extend(trace[post_norm_index+1:silu_index])

    # gemm_all_reduce and after_silu_gemm
    if has_all_reduce:
        gemm_all_reduce_end_index = all_reduce_end_indexes[-1]
        if match(trace[gemm_all_reduce_end_index-1],"Memcpy"):
            gemm_all_reduce_index = gemm_all_reduce_end_index-1
        else:
            gemm_all_reduce_index = gemm_all_reduce_end_index
        gemm_all_reduce.extend(trace[gemm_all_reduce_index: gemm_all_reduce_end_index+1])
        after_silu_gemm.extend(trace[silu_index+1: gemm_all_reduce_index])
    else:
        after_silu_gemm.extend(trace[silu_index+1:])

    # padding 
    padding(rms_norm, rms_norm_num)
    padding(qkv_proj, qkv_proj_num)
    padding(qk_norm, qk_norm_num)
    padding(rotary_emb, rotary_emb_num)
    padding(mha, mha_num)
    padding(o_proj, o_proj_num)
    padding(mha_all_reduce, mha_all_reduce_num)
    padding(post_norm, post_norm_num)
    padding(before_silu_gemm, before_silu_gemm_num)
    padding(silu, silu_num)
    padding(after_silu_gemm, after_silu_gemm_num)
    padding(gemm_all_reduce, gemm_all_reduce_num)

    # concat
    details = [rms_norm, qkv_proj, qk_norm, rotary_emb, mha, o_proj, 
              mha_all_reduce, post_norm, before_silu_gemm, silu, 
              after_silu_gemm, gemm_all_reduce, bubbles]
    
    return details


def print_details(details):
    kernels_dur, trace_dur = 0, 0
    # compute kernel time
    for item in details[:-1]:
        for n, t, ts in item:
            kernels_dur += t
    bubble_t = details[-1][0][1]
    trace_dur = kernels_dur + bubble_t

    # print with name
    for item in details:
        for n, t, ts in item:
            print(f'"{n[:60]}",  {t:.3f}')
    print(f'=> kernel total = {kernels_dur:.3f}')
    print(f'=> layer total = {trace_dur:.3f}')


def save_to_csv(data): 
    '''
              prefill_bs_1     rms_norm                mha
    data = [ (title, [ [(n,t,ts),(n,t,ts)], [(n,t,ts),(n,t,ts)] ]), 
             (title, [ [(n,t,ts),(n,t,ts)], [(n,t,ts),(n,t,ts)] ])]
    
    
              prefill_bs_1  
    data_sum = [ (title,       [ 1, 3, 5 ]), 
                 (title,       [ 1, 3, 5 ])]

                 prefill_bs_1     
    data_flat = [ (title, [ (n,t,ts),(n,t,ts), (n,t,ts),(n,t,ts)] ), 
                  (title, [ (n,t,ts),(n,t,ts), (n,t,ts),(n,t,ts)] )
    '''
    rows = []

    # create summary chart
    data_sum = list()
    for title, details in data:
        detail_sum = list()
        for i, part in enumerate(details):
            part_sum = sum([k[1] for k in part])
            detail_sum.append(part_sum)
            if i == 5:
                detail_sum.append(sum(detail_sum[1:6]))
        data_sum.append((title, detail_sum))
    
    rows.append(",".join([d[0] for d in data_sum]))
    rows_len = len(data_sum[0][1])
    for row_index in range(rows_len):
        row_str = ""
        for title, details_sum in data_sum:
            t = details_sum[row_index]
            row_str += f"{t:.3f},"
        rows.append(row_str.rstrip(","))


    # create details chart
    rows.extend([""]*3)
    data_flat = list()
    for title, details in data:
        details_flat = list()
        for d in details:
            details_flat.extend(d)
        data_flat.append((title, details_flat))

    rows.append(",".join([d[0] for d in data_flat]))
    rows_len = len(data_flat[0][1])
    for row_index in range(rows_len):
        row_str = ""
        for title, details_flat in data_flat:
            n,t,ts = details_flat[row_index]
            row_str += f"{t:.3f},"
        rows.append(row_str.rstrip(","))


    assert rows
    out_name = f"kernel.csv"
    with open(out_name, "w") as f:
        for row in rows:
            f.write(row + "\n")
    print(f" ==> Saved to {out_name}")
    
 

def coefficient_of_variation(data):
    mean = statistics.mean(data)
    std_dev = statistics.stdev(data) 
    cv = std_dev / mean
    return cv


def print_layer_traces(layer_traces):
    min_layer_trace = min(layer_traces)
    max_layer_trace = max(layer_traces)
    median_layer_trace = statistics.median(layer_traces)
    cv = coefficient_of_variation([t[0] for t in layer_traces])
    print("=======================================================================")
    print("[min details]")
    details = get_details(min_layer_trace)
    print_details(details)
    print(f"\n[min] kernel total = {min_layer_trace[1]:.3f}, layer total = {min_layer_trace[0]:.3f}")
    print(f"[median] kernel total = {median_layer_trace[1]:.3f}, layer total = {median_layer_trace[0]:.3f}")
    print(f"[max] kernel total = {max_layer_trace[1]:.3f}, layer total = {max_layer_trace[0]:.3f}")
    print(f'[variance] coefficient of variation = {cv:.3f} {" !!!This is large!!!" if cv >0.1 else "" }')
    print("=======================================================================\n")
    return details


def process_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    gpu_trace = get_gpu_trace(data['traceEvents'])
    forward_trace, start_index = get_one_forward_trace(gpu_trace)
    # WAR
    if not forward_trace:
        forward_trace = gpu_trace
    layer_traces = get_layer_traces(forward_trace)
    details = print_layer_traces(layer_traces)
    # aggressively decision
    is_prefill = details[0][0][1] > 50
    prefill_details, decode_details = [], []

    # if no prefill, return decode now
    if not is_prefill:
        decode_details = details
        return [], decode_details
    # if prefill is found, then try find decode 
    else:
        prefill_details = details 
        if start_index is not None:
            forward_trace, start_index = get_one_forward_trace(gpu_trace, start_index=start_index)
            layer_traces = get_layer_traces(forward_trace)
            decode_details = print_layer_traces(layer_traces)
    return prefill_details, decode_details
    

if __name__ == '__main__':
    json_file_list = sys.argv[1:]
    json_file_list = [ (int(re.search(r'_b(\d+)_', json_file).group(1)), json_file) for json_file in json_file_list ]
    json_file_list.sort()
    data = list()
    for bs, json_file in json_file_list:
        print(f"==== batch size [{bs}]==")
        prefill_details, decode_details = process_json_file(json_file)
        if bs == 1 and prefill_details:
            data.append((f"Prefill_BS_{bs}", prefill_details))
        if decode_details:
                data.append((f"Decode_BS_{bs}", decode_details))

    save_to_csv(data)
