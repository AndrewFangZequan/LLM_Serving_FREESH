# LLM Serving: LLMCare

> A modular **routing + scheduling** system for LLM inference that jointly optimizes **energy/carbon**, **SLOs** across heterogeneous, geo-distributed GPU clusters.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

## Highlights

- **Three-layer co-design**
  - **Pool-level (slow timescale)**: MILP-based optimizer for cross-region placement, request partitioning, and tensor-parallel (TP) mode selection with switchable **energy** or **carbon** objectives.
  - **GPU-level (seconds)**: **MIAD (Multiplicative Increase, Additive Decrease)** dynamic frequency control to reduce power while meeting SLOs.
- **Five “control knobs”**: site selection, partitioning, request type, execution mode (TP / model-GPU pairing), frequency (DVFS/MIAD).
- **Carbon-intensity aware routing**: leverages spatiotemporal grid carbon signals to shift work to cleaner regions/periods.
- **Plug-and-play**: integrates with common inference runtimes (e.g., vLLM) and multiple solvers (e.g., Gurobi / open-source backends).

