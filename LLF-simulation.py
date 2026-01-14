"""
Comparison of Four Algorithms (Simulation)

This script implements and compares four scheduling policies:
- FCFS (First-Come First-Served)
- SRTF (Shortest Remaining Time First)
- EDF (Earliest Deadline First)
- LLF (Least Laxity First)

Metrics reported:
- SLO violation rate (Breach Rate)
- Average time-to-first-token (Average TTFT)
- Long-term service fairness (Fairness)
- Average maximum waiting time (Average Max Waiting Time)
"""

import os
import json
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ===================== Fixed experiment configuration =====================

# Scheduling policies under comparison
ALGORITHMS = ["FCFS", "SRTF", "EDF", "LLF"]

# QPS values to sweep
QPS_LIST = [5, 10, 15, 20, 25]

# Total number of requests (fixed)
NUM_REQUESTS = 100

# Number of parallel workers 
NUM_WORKERS = 20

RANDOM_SEED = 42

SIMULATION_STEP = 0.1


# ===================== Scheduling and SLO parameters =====================

# Scheduling window coefficient for EDF / LLF
WINDOW_ALPHA = 1.4

# Threshold coefficient for SLO violation
BREACH_ALPHA = 5


# ===================== Workload dataset paths =====================

DATASET_PATHS = {
    "SS": "prompts_short_answers_short.jsonl",
    "SL": "prompts_short_answers_long.jsonl",
    "LS": "prompts_long_answers_short.jsonl",
    "LL": "prompts_long_answers_long.jsonl",
}

WORKLOAD_PARAMS = {
    "SL": {"ttft_ms": 125.9, "tbt_ms": 27.61, "label": "Short Prompt, Long Response (SL)"},
    "LL": {"ttft_ms": 77.99,  "tbt_ms": 26.27, "label": "Long Prompt, Long Response (LL)"},
    "LS": {"ttft_ms": 140.69, "tbt_ms": 37.87, "label": "Long Prompt, Short Response (LS)"},
    "SS": {"ttft_ms": 144.49, "tbt_ms": 27.34, "label": "Short Prompt, Short Response (SS)"},
}

# Order of workloads for reporting
WORKLOAD_ORDER = ["SL", "LL", "LS", "SS"]


# ===================== Fairness configuration =====================

# Number of logical clients used for fairness accounting
K_CLIENTS = 4


# ===================== Utility functions =====================

def load_prompts_and_tokens(jsonl_path, n_needed, seed=12345):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "prompt" in obj:
                    samples.append(obj)
            except Exception:
                pass

    if not samples:
        raise RuntimeError(f"No valid samples found in dataset: {jsonl_path}")

    rng = random.Random(seed)
    out_toks = []
    for _ in range(n_needed):
        obj = rng.choice(samples)
        ov = None
        for k in ["output_token_count", "output_tokens", "completion_tokens", "n_output_tokens"]:
            if k in obj and obj[k] is not None:
                try:
                    ov = int(obj[k])
                    break
                except Exception:
                    pass
        if ov is None:
            ov = 128
        out_toks.append(ov)

    return [], [], out_toks



def gen_arrivals_fixed_qps(qps, n_requests, seed=1234):
    rng = random.Random(seed)
    t = 0.0
    arr = []
    while len(arr) < n_requests:
        t += rng.expovariate(qps)
        arr.append(t)
    # Normalize start time to 0
    base = arr[0]
    return [a - base for a in arr]


def gen_jobs_for(qps, n_requests, workload, seed):
    arrivals = gen_arrivals_fixed_qps(qps, n_requests, seed)
    dataset_path = DATASET_PATHS.get(workload, "__fallback__")
    _, _, out_toks = load_prompts_and_tokens(dataset_path, len(arrivals), seed + 12345)
    return [(arrivals[i], i + 1, out_toks[i]) for i in range(len(arrivals))]


# ===================== Job state definition =====================

@dataclass
class JobState:
    arrival: float
    jid: int
    total: int
    remaining: int
    token_budget: float = 0.0
    first_token_finished_at: Optional[float] = None
    finished_tokens_at: List[float] = field(default_factory=list)


# ===================== Core scheduling simulation =====================

def simulate(policy, jobs, num_workers, tokens_per_sec, ttft_seconds):
    # Helper: Predicted latency for SLO calculation
    def predicted_latency(total_tokens):
        return ttft_seconds + total_tokens / tokens_per_sec

    # Helper: Internal scheduling deadline
    def sched_window(total_tokens):
        return WINDOW_ALPHA * predicted_latency(total_tokens)

    # Helper: SLO breach threshold
    def breach_slo(total_tokens):
        return BREACH_ALPHA * predicted_latency(total_tokens)

    # Note: We do NOT ceil arrivals anymore to preserve precision
    arrivals_map = {jid: arr for (arr, jid, _) in jobs}
    totals_map = {jid: tot for (_, jid, tot) in jobs}

    # Initialize Job States
    J = {}
    for arr, jid, tot in jobs:
        J[jid] = JobState(
            arrival=arr,
            jid=jid,
            total=tot,
            remaining=tot
        )

    def client_of(jid):
        # Map Job ID to a logical client ID (Round Robin)
        return (jid - 1) % K_CLIENTS

    # Fairness accounting variables
    WQ = 2.0
    served_total = {c: 0.0 for c in range(K_CLIENTS)}
    served_epoch = {c: 0.0 for c in range(K_CLIENTS)}
    prev_backlogged = None
    max_gap = 0.0

    t = 0.0
    
    tokens_per_step = tokens_per_sec * SIMULATION_STEP

    def all_done():
        return all(st.remaining <= 0 for st in J.values())

    # Main Simulation Loop
    while not all_done():
        # Identify "Alive" jobs: Arrived (arrival <= t) AND not finished (remaining > 0)
        alive = [jid for jid, st in J.items() if st.arrival <= t and st.remaining > 0]

        chosen = []
        if alive:
            # Dynamic attributes for sorting
            def deadline(jid):
                return J[jid].arrival + sched_window(totals_map[jid])

            if policy == "FCFS":
                # Static priority: Arrival time
                chosen = sorted(alive, key=lambda j: (J[j].arrival, j))[:num_workers]
            
            elif policy == "SRTF":
                # Dynamic priority: Remaining tokens
                chosen = sorted(alive, key=lambda j: (J[j].remaining, j))[:num_workers]
            
            elif policy == "EDF":
                # Static priority: Deadline
                chosen = sorted(alive, key=lambda j: deadline(j))[:num_workers]
            
            elif policy == "LLF":
                # Dynamic priority: Laxity
                # Laxity = (Deadline - CurrentTime) - (Remaining / Rate)
                # Updated every SIMULATION_STEP
                def laxity(j):
                    execution_time_needed = J[j].remaining / tokens_per_sec
                    return deadline(j) - t - execution_time_needed
                
                chosen = sorted(alive, key=lambda j: laxity(j))[:num_workers]

        # --- Fairness Logic ---
        # Track which clients have active jobs in this step
        current_backlogged = frozenset(client_of(jid) for jid in alive)
        
        if current_backlogged != prev_backlogged:
            served_epoch = {c: 0.0 for c in range(K_CLIENTS)}
            prev_backlogged = current_backlogged

        produced_by_client = {c: 0.0 for c in range(K_CLIENTS)}

        # --- Execution Logic ---
        for jid in chosen:
            st = J[jid]
            
            # Add capacity to the job's budget
            st.token_budget += tokens_per_step
            
            # Consume whole tokens from budget
            # We treat token generation as integer steps for recording
            spend = min(st.remaining, int(st.token_budget))
            
            if spend > 0:
                # Record completion times for these tokens
                # We assume they finished evenly distributed or at end of step
                # For simplicity, mark them as finished at current t
                for _ in range(spend):
                    st.finished_tokens_at.append(t)
                    if st.first_token_finished_at is None:
                        st.first_token_finished_at = t
                
                st.remaining -= spend
                st.token_budget -= spend
                produced_by_client[client_of(jid)] += spend

        # Update Fairness Stats
        for c in range(K_CLIENTS):
            inc = WQ * produced_by_client[c]
            served_total[c] += inc
            if c in current_backlogged:
                served_epoch[c] += inc

        # Calculate Gap for this epoch
        if len(current_backlogged) >= 2:
            vals = [served_epoch[c] for c in current_backlogged]
            gap = max(vals) - min(vals)
            max_gap = max(max_gap, gap)

        # Advance Time
        t += SIMULATION_STEP

    # ===================== Metric Computation =====================

    # 1. Average TTFT
    ttfts = [
        st.first_token_finished_at - st.arrival
        for st in J.values()
        if st.first_token_finished_at is not None
    ]
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0

    # 2. Average Maximum Waiting Time (Inter-token latency spikes)
    max_waits = []
    for st in J.values():
        if not st.finished_tokens_at:
            continue
        # Gap between tokens
        gaps = [
            st.finished_tokens_at[i] - st.finished_tokens_at[i - 1]
            for i in range(1, len(st.finished_tokens_at))
        ]
        max_gap_t = max(gaps) if gaps else 0.0
        # TTFT is also a "wait"
        ttft_j = st.first_token_finished_at - st.arrival
        max_waits.append(max(ttft_j, max_gap_t))

    avg_max_waiting = sum(max_waits) / len(max_waits) if max_waits else 0.0

    # 3. Breach Rate
    breach = 0
    for jid, st in J.items():
        slo = breach_slo(st.total)
        finish = max(st.finished_tokens_at) if st.finished_tokens_at else None
        
        # If job never finished (shouldn't happen) or finished late
        if finish is None or (finish - st.arrival) > slo:
            breach += 1

    breach_rate = breach / len(J)

    # 4. Fairness (Jain's Index-like derived from Gap)
    avg_service = sum(served_total.values()) / K_CLIENTS
    # Gap-based fairness metric: 1.0 is perfect, <1.0 is worse
    fairness = 1.0 if avg_service <= 1e-9 else max(0.0, 1.0 - max_gap / avg_service)

    return {
        "breach_rate": breach_rate,
        "avg_ttft": avg_ttft,
        "fairness": fairness,
        "avg_max_waiting": avg_max_waiting,
    }


if __name__ == "__main__":
    print(f"Starting simulation with Time Step = {SIMULATION_STEP}s")
    
    for workload in WORKLOAD_ORDER:
        params = WORKLOAD_PARAMS[workload]
        ttft_s = params["ttft_ms"] / 1000.0
        tps = 1000.0 / params["tbt_ms"]

        print(f"\n===== Workload: {workload} ({params['label']}) =====")

        for qps in QPS_LIST:
            jobs = gen_jobs_for(qps, NUM_REQUESTS, workload, RANDOM_SEED + int(qps * 10))
            print(f"\nQPS = {qps}")
            print(f"{'Algo':<6} | {'Breach':<12} | {'TTFT (s)':<12} | {'Fairness':<10} | {'MaxWait (s)':<12}")
            print("-" * 65)
            
            for algo in ALGORITHMS:
                r = simulate(algo, jobs, NUM_WORKERS, tps, ttft_s)
                print(
                    f"{algo:<6} | "
                    f"{r['breach_rate']:.3f}        | "
                    f"{r['avg_ttft']:.3f}        | "
                    f"{r['fairness']:.3f}      | "
                    f"{r['avg_max_waiting']:.3f}"
                )