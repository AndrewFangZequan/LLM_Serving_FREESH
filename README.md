# FREESH: Fair, Resource- and Energy-Efficient Scheduling for LLM Serving on Heterogeneous GPUs

> A modular **routing + scheduling** system for LLM inference that jointly optimizes **energy/carbon**, **SLOs**, and **fairness** across heterogeneous, geo-distributed GPU clusters.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![arXiv](https://img.shields.io/badge/arXiv-2511.00807-b31b1b.svg)](https://arxiv.org/abs/2511.00807)

## Highlights

- **Three-layer co-design**
  - **Pool-level (slow timescale)**: MILP-based optimizer for cross-region placement, request partitioning, and tensor-parallel (TP) mode selection with switchable **energy** or **carbon** objectives.
  - **GPU-level (seconds)**: **MIAD (Multiplicative Increase, Additive Decrease)** dynamic frequency control to reduce power while meeting SLOs.
  - **Request-level (sub-second)**: **LLF (Least Laxity First)** preemptive scheduler for fairness and lower tail latency (reduces FCFS head-of-line blocking).
- **Six “control knobs”**: site selection, partitioning, request type, execution mode (TP / model-GPU pairing), frequency (DVFS/MIAD), and scheduling policy (LLF).
- **Carbon-intensity aware routing**: leverages spatiotemporal grid carbon signals to shift work to cleaner regions/periods.
- **Plug-and-play**: integrates with common inference runtimes (e.g., vLLM) and multiple solvers (e.g., Gurobi / open-source backends).

> Full method and evaluations: **https://arxiv.org/abs/2511.00807**
