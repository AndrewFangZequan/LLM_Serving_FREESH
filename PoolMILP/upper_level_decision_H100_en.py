import pulp
import math
import numpy as np
import pandas as pd
from statistics import mean

def initialize_sets(num_locations, num_requests, num_modes, num_gpus, num_times, Q):
    """
    Initializes all sets and identifiers.

    Args:
        1.num_locations: The number of data center locations.
        2.num_requests[key]: The number of request types for each partition way.
        3.num_modes: The number of working modes. Each mode can have custom parallel/split configurations 
                   and corresponding GPU counts. Currently, only parallelism on the same GPU type is considered.
        4.num_gpus: The number of GPU types (e.g., 2 for A100 and H100).
        5.num_times: The number of time steps. The current time window is 30 minutes, with 6 steps of 5 minutes each.
        6. Q: A set of partition way (e.g., [['a'], ['b']]). The number of requests can differ for different partition ways.
    """
    N = [f"loc{i+1}" for i in range(num_locations)]    # Data center location identifiers: loc1, loc2, ...
    R = {}  # Dictionary to hold request types for each quantile
    for quantiles in Q:
        key = str(quantiles)
        R[key] = [f"req{i+1}" for i in range(num_requests[key])]
    M = [f"mode{i+1}" for i in range(num_modes)]       # Set of working modes
    C = [f"gpu{i+1}" for i in range(num_gpus)]        # Set of GPU types
    T = [f"time{i+1}" for i in range(num_times)]       # Set of time steps
    return N, R, M, C, T

def generate_parameters(N, R, M, C, T, problem_idx=0, beta=1.0):
    """
    Generates all parameter dictionaries, including power, throughput, latency, etc.

    Returns:
        e: Carbon emission coefficient for each location at different times. (n, t) -> float
        P_tilde: Power consumption for requests in each mode under different quantiles. (key, r, m) -> float
        L_bar: Upper limit of throughput (QPS) for requests in each mode under different quantiles. (key, r, m) -> float
        L_tilde: Predicted QPS demand for requests at each time step under different quantiles. (key, r, t) -> float
        G: GPU requirement of each mode for different GPU types. (m, c) -> int
        G_bar: Total number of c-type GPUs at each location. (n, c) -> int
        D: Latency for requests in each mode under different quantiles. (key, r, m) -> float
        D_bar: Upper limit of latency for requests under different quantiles. (key, r) -> float
        TBT: Time Between Tokens for requests in each mode under different quantiles. (key, r, m) -> float
        TBT_bar: Upper limit of TBT for requests under different quantiles. (key, r) -> float
        TTFT: Time To First Token for requests in each mode under different quantiles. (key, r, m) -> float
        TTFT_bar: Upper limit of TTFT for requests under different quantiles. (key, r) -> float
        e_instance: Raw carbon emission data.
        base_throughput_instance: Raw base throughput data.
    """
    import random
    # GPU requirement matrix for modes: The first 3 modes use 2/4/8 A100s, the last one uses 4 H100s.
    G_m_c_instance = [[2,0],[4,0],[8,0], [0,4]]
    # GPU capacity limit for each location.
    G_bar_instance = [[4,4], [12,2], [8,0]]
    
    # Throughput upper limit (L_bar) data.
    L_bar_instance = {'[a]':[[5.49, 11.25, 14.85, 16.38], # SS 
                             [0, 2.52, 5.31, 2.07], # SL
                             [2.61, 6.93, 11.97, 6.93], # LS
                             [0, 1.44, 5.04, 0]], # LL
                      '[b]':[[20.25, 23.85, 23.85, 29.16], # SS
                             [0.99, 3.33, 6.75, 2.97], # SL
                             [6.3, 19.89, 19.17, 16.38], # LS
                             [1.26, 1.98, 4.68, 1.53]]}  # LL
    
    # Power consumption data for each partition way: P_tilde_instance[key][req][mode]
    # The first 3 modes use 2/4/8 A100s, the last one uses 4 H100s.
    P_tilde_instance = {'[a]':[[745.5996923, 1466.284, 2422.94, 1295.21], # SS
                               [698.661791, 1231.476984, 2149.86, 1020.23], # SL
                               [747.2272727, 1472.297377, 2688.76, 1261.54], # LS
                               [709.9641791, 1285.822419, 2201.83, 971.77]], # LL
                        '[b]':[[796.233, 1471.44, 2204.20, 1260.13], # SS
                               [739.2547059, 1235.504516, 2234.967551, 1039.82], # SL
                               [772.5930303, 1524.966607, 2577.1718, 1309.87], # LS
                               [751.4352941, 1267.916935, 2258.218235, 1011.11]]} # LL
    
    # Load base throughput data from xlsx files
    df_a = pd.read_excel('base_throughput_a.xlsx')
    df_b = pd.read_excel('base_throughput_b.xlsx')
    
    # Get data for current problem
    start_idx = problem_idx * 6
    end_idx = start_idx + 6
    
    # Create base_throughput_instance from dataframes with scaling factor beta
    # Scale the base throughput by beta
    base_throughput_instance = {
        '[a]': [[val/beta for val in df_a.iloc[start_idx:end_idx, i].tolist()] for i in range(4)],
        '[b]': [[val/beta for val in df_b.iloc[start_idx:end_idx, i].tolist()] for i in range(4)]
    }
    
    # Load carbon emission data from csv
    e_df = pd.read_csv('e_instance.csv')
    e_instance = e_df.iloc[start_idx:end_idx, :3].values.T.tolist()
    
    # delay_instance[key][req][mode]: latency of request 'req' using mode 'mode' under quantile 'key'.
    delay_instance = {'[a]':[[8.85, 7.746, 5.745, 2.86], # SS
                             [44.163, 25.36, 18.476,16.82], # SL
                             [7.027, 8.448, 5.711, 2.89], # LS
                             [30.997,25.095, 19.142, 16.726]], # LL
                      '[b]':[[7.405, 3.327, 2.088, 8.021], # SS
                             [27.314,21.25, 15.242, 32.057], # SL
                             [6.328, 8.664, 3.089, 11.478], # LS
                             [27.919,22.082, 15.151, 42.760]]} # LL
    
    latency_bar_scale = 5
    thres = 10000
    delay_ss_a = mean(filter(lambda k: k <= thres, delay_instance['[a]'][0]))*latency_bar_scale
    delay_sl_a = mean(filter(lambda k: k <= thres, delay_instance['[a]'][1]))*latency_bar_scale
    delay_ls_a = mean(filter(lambda k: k <= thres, delay_instance['[a]'][2]))*latency_bar_scale
    delay_ll_a = mean(filter(lambda k: k <= thres, delay_instance['[a]'][3]))*latency_bar_scale
    delay_ss_b = mean(filter(lambda k: k <= thres, delay_instance['[b]'][0]))*latency_bar_scale
    delay_sl_b = mean(filter(lambda k: k <= thres, delay_instance['[b]'][1]))*latency_bar_scale
    delay_ls_b = mean(filter(lambda k: k <= thres, delay_instance['[b]'][2]))*latency_bar_scale
    delay_ll_b = mean(filter(lambda k: k <= thres, delay_instance['[b]'][3]))*latency_bar_scale

    # base_delay_instance[key][req]: latency upper limit for request 'req' under quantile 'key'.
    base_delay_instance = {'[a]':[delay_ss_a, delay_sl_a, delay_ls_a, delay_ll_a], 
                           '[b]':[ delay_ss_b, delay_sl_b, delay_ls_b, delay_ll_b]}
    print(base_delay_instance)
    
    # TBT_instance[key][req][mode]: Time Between Tokens for request 'req' using mode 'mode' under quantile 'key'.
    TBT_instance = {'[a]':[[59.60, 46.773, 33.70, 2], # SS, 
                           [48.98, 37.701, 26.30,29.09], # SL
                           [67.85, 56.691, 43.188,37.93], # LS
                           [48.98, 35.22, 27.155, 28.41]], # LL
                    '[b]':[[103.56, 45.20, 23.34, 31.08], # SS
                           [51.34, 38.56848286, 27.61, 30.19], # SL
                           [78.59, 100.0185784, 37.82, 35.83], # LS
                           [53.47, 35.16418645, 26.27, 29.67]]} # LL
    
    # base_TBT_instance[key][req]: TBT upper limit for request 'req' under quantile 'key'.
    base_TBT_instance = {'[a]':[150, 150, 150, 150], '[b]':[150, 150, 150, 150]}
    
    # TTFT_instance[key][req][mode]: Time To First Token for request 'req' using mode 'mode' under quantile 'key'.
    TTFT_instance = {'[a]':[[120.36, 124.02, 226.24, 76.35], # SS
                            [68.28, 73.224, 76.61, 58.25], # SL
                            [195.58, 178.0074, 162.97, 405.31], # LS
                            [151.18,120.38, 93.76, 86.61]], # LL
                     '[b]':[[225.1779768, 127.50, 144.49, 59.45], # SS
                            [107.3190549, 72.0319532, 125.89, 58.72], # SL
                            [168.3436671, 616.3113967, 140.69, 71.73], # LS
                            [140.3118727,82.90107076, 77.89, 63.70]]} # LL

    # base_TTFT_instance[key][req]: TTFT upper limit for request 'req' under quantile 'key' (note: all in milliseconds).
    base_TTFT_instance = {'[a]':[300, 300, 600, 600], '[b]':[300, 300, 600, 600]}
    
    # Extracts the index number from the set name (e.g., 1 from 'loc1', 2 from 'req2').
    # Used to map named identifiers to array indices for various types like loc/req/mode/gpu/time.
    def get_index(name):
        return int(name.split('loc')[-1].split('req')[-1].split('mode')[-1].split('gpu')[-1].split('time')[-1]) - 1
    
    # Build parameter dictionaries: Convert raw data arrays into dictionaries with entity identifiers as keys.
    # Carbon emission coefficient dictionary: key is (location, time), value is the corresponding coefficient.
    e = {(n, t): e_instance[get_index(n)][get_index(t)]
         for n in N for t in T}
    print(e)
    # Power consumption dictionary: key is (quantile_key, request, mode), value is the power consumption.
    P_tilde = {}
    for key in R:
        for r in R[key]:
            for m in M:
                P_tilde[(key, r, m)] = P_tilde_instance[key][get_index(r)][get_index(m)]
    
    # QPS upper limit dictionary: key is (quantile_key, request, mode), value is the throughput upper limit.
    L_bar = {}
    for key in R:
        for r in R[key]:
            for m in M:
                L_bar[(key, r, m)] = L_bar_instance[key][get_index(r)][get_index(m)]
    
    # Predicted QPS demand dictionary: key is (quantile_key, request, time), value is the base throughput demand.
    L_tilde = {}
    for key in R:
        for r in R[key]:
            for t in T:
                L_tilde[(key, r, t)] = base_throughput_instance[key][get_index(r)][get_index(t)]
    
    # GPU requirement dictionary: key is (mode, GPU type), value is the number of GPUs required.
    G = {(m, c): G_m_c_instance[get_index(m)][get_index(c)]
         for m in M for c in C}
    
    # GPU capacity dictionary: key is (location, GPU type), value is the number of available GPUs.
    G_bar = {(n, c): G_bar_instance[get_index(n)][get_index(c)]
             for n in N for c in C}
    
    # Latency dictionary: key is (quantile_key, request, mode), value is the latency.
    D = {}
    for key in R:
        for r in R[key]:
            for m in M:
                D[(key, r, m)] = delay_instance[key][get_index(r)][get_index(m)]
    
    # Latency upper limit dictionary: key is (quantile_key, request), value is the latency upper limit.
    D_bar = {}
    for key in R:
        for r in R[key]:
            D_bar[(key, r)] = base_delay_instance[key][get_index(r)]
    
    # TBT dictionary: key is (quantile_key, request, mode), value is the TBT.
    TBT = {}
    for key in R:
        for r in R[key]:
            for m in M:
                TBT[(key, r, m)] = TBT_instance[key][get_index(r)][get_index(m)]
                
    # TBT upper limit dictionary: key is (quantile_key, request), value is the TBT upper limit.
    TBT_bar = {}
    for key in R:
        for r in R[key]:
            TBT_bar[(key, r)] = base_TBT_instance[key][get_index(r)]
    
    # TTFT dictionary: key is (quantile_key, request, mode), value is the TTFT.
    TTFT = {}
    for key in R:
        for r in R[key]:
            for m in M:
                TTFT[(key, r, m)] = TTFT_instance[key][get_index(r)][get_index(m)]

    # TTFT upper limit dictionary: key is (quantile_key, request), value is the TTFT upper limit.
    TTFT_bar = {}
    for key in R:
        for r in R[key]:
            TTFT_bar[(key, r)] = base_TTFT_instance[key][get_index(r)]

    return e, P_tilde, L_bar, L_tilde, G, G_bar, D, D_bar, TBT, TBT_bar, TTFT, TTFT_bar, e_instance, base_throughput_instance

def solve_problem(problem_idx, beta=1.0, count = 0):  # Add beta parameter to scale base_throughput
    print(f"\nProblem {problem_idx} details:")
    # Set basic parameters for the optimization problem
    num_locations = 3  # Number of data center locations
    num_requests = {'[a]':4, '[b]':4}  # Number of request types for each quantile
    num_modes = 4      # Parallelism configuration
    num_gpus = 2       # GPU types: A100 and H100
    num_times = 6      # Number of time slots for calculating carbon emissions

    # Quantile set: used for optimal request classification
    Q = ['[a]']

    # Initialize all sets: location, request, mode, GPU type, time step
    N, R, M, C, T = initialize_sets(num_locations, num_requests, num_modes, num_gpus, num_times, Q)

    # Generate all necessary parameters for the optimization problem
    e, P_tilde, L_bar, L_tilde, G, G_bar, D, D_bar, TBT, TBT_bar, TTFT, TTFT_bar, e_instance, base_throughput_instance = generate_parameters(N, R, M, C, T, problem_idx, beta)
    alpha = 1.1    # Throughput demand scaling factor to ensure sufficient margin
    delta_t = 300   # Time interval (seconds) for calculating total energy consumption

    # Create the problem
    problem = pulp.LpProblem("CarbonEmissionMinimization", pulp.LpMinimize)

    # Define decision variables for the optimization problem
    x = {}  # x[q_idx][(n,r,m)]: number of instances of mode m for request r at location n under quantile q_idx
    y = pulp.LpVariable.dicts("y", range(len(Q)), cat=pulp.LpBinary)  # y[q_idx]=1 if quantile is selected, 0 otherwise

    for q_idx, quantiles in enumerate(Q):  # Create resource allocation variables for each quantile
        key = str(quantiles)
        x[q_idx] = pulp.LpVariable.dicts(
            f"x_{q_idx}", 
            [(n, r, m) for n in N for r in R[key] for m in M],  # All possible (location, request, mode) combinations
            lowBound=0,  # The allocated quantity must be non-negative
            cat=pulp.LpInteger  # The variable must be an integer
        )

    # Objective function: Minimize the total carbon emissions over all time steps
    objective = pulp.lpSum(
        e[(n, t)] * x[q_idx][(n, r, m)] * P_tilde[(str(quantiles), r, m)] * delta_t
        for q_idx, quantiles in enumerate(Q)
        for n in N 
        for r in R[str(quantiles)]
        for m in M
        for t in T
    )
    problem += objective
    problem.sense = pulp.LpMinimize

    # Constraint 1: Only one quantile combination can be selected
    problem += pulp.lpSum(y[q_idx] for q_idx in range(len(Q))) == 1

    # Constraint 2: Link resource allocation with quantile selection
    M_large = 10000  
    for q_idx, quantiles in enumerate(Q):
        key = str(quantiles)
        for n in N:
            for r in R[key]:
                for m in M:
                    problem += x[q_idx][(n, r, m)] <= M_large * y[q_idx]

    # Constraint 3: QPS constraint
    for q_idx, quantiles in enumerate(Q):
        key = str(quantiles)
        for r in R[key]:
            for t in T:
                problem += pulp.lpSum(
                    x[q_idx][(n, r, m)] * L_bar[(key, r, m)]
                    for n in N for m in M
                ) >= alpha * L_tilde[(key, r, t)] * y[q_idx]

    # Constraint 4: GPU resource constraint
    for n in N:
        for c in C:
            problem += pulp.lpSum(
                x[q_idx][(n, r, m)] * G[(m, c)]
                for q_idx, quantiles in enumerate(Q)
                for r in R[str(quantiles)]
                for m in M
            ) <= G_bar[(n, c)]

    # Constraint 5: Latency constraint
    for q_idx, quantiles in enumerate(Q):
        key = str(quantiles)
        for n in N:
            for r in R[key]:
                for m in M:
                    problem += x[q_idx][(n, r, m)] * D[(key, r, m)] <= x[q_idx][(n, r, m)] * D_bar[(key, r)]

    # Constraint 6: TBT constraint
    for q_idx, quantiles in enumerate(Q):
        key = str(quantiles)
        for n in N:
            for r in R[key]:
                for m in M:
                    problem += x[q_idx][(n, r, m)] * TBT[(key, r, m)] <= x[q_idx][(n, r, m)] * TBT_bar[(key, r)]

    # Constraint 7: TTFT constraint
    for q_idx, quantiles in enumerate(Q):
        key = str(quantiles)
        for n in N:
            for r in R[key]:
                for m in M:
                    problem += x[q_idx][(n, r, m)] * TTFT[(key, r, m)] <= x[q_idx][(n, r, m)] * TTFT_bar[(key, r)]

    # Solve the problem
    problem.solve()

    results = []
    print(e_instance)
    # Get e_instance values for all locations at this time step
    e_vals = [e_instance[i][:] for i in range(3)]
    # Get base_throughput values for this time step
    base_a = [base_throughput_instance['[a]'][i][:] for i in range(4)]
    base_b = [base_throughput_instance['[b]'][i][:] for i in range(4)]
    
    if problem.status == pulp.LpStatusOptimal:
        obj_value = problem.objective.value()
        # Print objective components for negative values
        if obj_value < 0:
            print(f"\nNegative objective value found: {obj_value}")
            print("Objective components:")
            for q_idx, quantiles in enumerate(Q):
                if y[q_idx].varValue > 0.5:  # Only check the selected quantile
                    key = str(quantiles)
                    for n in N:
                        for r in R[key]:
                            for m in M:
                                for t in T:
                                    if x[q_idx][(n, r, m)].varValue > 0:
                                        component = e[(n, t)] * x[q_idx][(n, r, m)].varValue * P_tilde[(key, r, m)] * delta_t
                                        print(f"Location {n}, Request {r}, Mode {m}, Time {t}:")
                                        print(f"  e[{n},{t}] = {e[(n, t)]}")
                                        print(f"  x[{n},{r},{m}] = {x[q_idx][(n, r, m)].varValue}")
                                        print(f"  P_tilde[{key},{r},{m}] = {P_tilde[(key, r, m)]}")
                                        print(f"  delta_t = {delta_t}")
                                        print(f"  Component value = {component}")
        # Record allocation results
        for q_idx, quantiles in enumerate(Q):
            if y[q_idx].varValue > 0.5:
                key = str(quantiles)
                for n in N:
                    for r in R[key]:
                        for m in M:
                            if x[q_idx][(n, r, m)].varValue > 0:
                                results.append((n, r, m, problem_idx, y[q_idx], x[q_idx][(n, r, m)].varValue, problem.objective.value(),
                                             P_tilde[(key, r, m)], e_vals[0], e_vals[1], e_vals[2], 
                                             base_a[0], base_a[1], base_a[2], base_a[3],
                                             base_b[0], base_b[1], base_b[2], base_b[3]))
    else:
        # Record the case of no solution
        results.append(("no_solution", "no_solution", "no_solution", problem_idx, -1, -1, -1,
                       e_vals[0], e_vals[1], e_vals[2],
                       base_a[0], base_a[1], base_a[2], base_a[3],
                       base_b[0], base_b[1], base_b[2], base_b[3]))

    return results

def run_simulation(beta=1.0):  # Add beta parameter
    all_results = []
    count = -1
    for i in range(48):  # 48 problems
        if not (i % 6):
            count += 1
        print(f"Solving problem {i+1}/48...")
        results = solve_problem(i, beta, count)
        all_results.extend(results)
    
    # Save results to a CSV file
    df = pd.DataFrame(all_results, columns=['location', 'request', 'mode', 'time_step', 'quntantile', 'value', 'objective', 'Power',
                                          'e_loc1', 'e_loc2', 'e_loc3',
                                          'base_a1', 'base_a2', 'base_a3', 'base_a4',
                                          'base_b1', 'base_b2', 'base_b3', 'base_b4'])
    df.to_csv(f'resource_allocation_results_beta{beta}.csv', index=False)
    print(f"Results saved to resource_allocation_results_beta_{beta}.csv")

if __name__ == "__main__":
    # Set the throughput scaling factor
    beta = 2
    run_simulation(beta)
