# Cold TP-switch experiment

Generate four request workloads, predict response lengths, route requests to vLLM
workers, and measure a cold tensor-parallel deployment switch with Monitor and
optional MIAD frequency control. **This experiment does not implement LLF scheduling
or invoke the pool-level MILP optimizer.**

By default, the experiment runs for **3600 logical seconds** and switches at **1800**.
At the boundary it freezes the clock, arrivals and dispatch, preserves unfinished
requests, and stops phase-1 monitoring. Manual GPU preparation is excluded from
experiment time. After readiness is signalled, the clock/arrivals resume, startup
power is recorded, and new vLLM workers start and join independently as they become
healthy. Retries restart from the original prompt and reuse their prediction.
At the duration limit, unfinished requests are discarded without an extra drain period.

## 1. Install and obtain weights

Use Python 3.10+. Install the Router dependencies:

```bash
cd LLM_Serving_FREESH/Cold_Switch
python -m pip install -r requirements.txt
```

On GPU hosts, install a compatible vLLM/CUDA environment and the target LLM. Monitor
requires `pandas`, `requests` and `nvidia-smi`; MIAD needs permission to set GPU clocks.

Download **`best_model.pt` from this repository's Releases** and place it beside the
existing tokenizer files in `Bert_Classifier/bert_llama3_70B/`. Missing weights cause
an explicit startup error. The predictor also loads `xlm-roberta-base` through
Transformers, so that base model must be cached or reachable.

## 2. Configure workers and traffic

Edit the example configuration files:

| File | Purpose |
| --- | --- |
| `configs/phase1.json`, `configs/phase2.json` | Worker addresses, GPU/TP mapping, weights, Monitor URLs and start/stop commands |
| `configs/qps.json` | Four source-workload QPS arrays; defaults to 12 segments, each at 1 QPS |
| `configs/controller.json` | Router URL, health/command timeouts, polling intervals and optional `ready_file` |

The example uses one host per request type per phase, TP2 in phase 1 and TP4 in
phase 2. Replace **all `.example.org` hosts** and adapt GPU counts and ports to your
hardware. Each phase must cover all four predicted request types, including types
whose source QPS is zero. Each worker needs an independent Monitor process/URL;
`gpu_ids` must match `CUDA_VISIBLE_DEVICES` and Monitor `--gpu-ids`.

Phase-1 vLLM workers can be started manually or via `start_commands` / per-worker
`start_command`. Phase 2 uses per-worker commands that launch services and return
promptly. Commands are argument arrays, not shell strings. The sample uses SSH and
`sudo -n systemctl start`; configure noninteractive SSH/sudo and model access first.
Edit `configs/vllm.service.example` and install the corresponding service on each
phase-2 host, then run `sudo systemctl daemon-reload`. Do not start new phase-2
vLLM workers before the readiness signal. Replace these commands if using another
service manager. `stop_commands` default to empty, so service shutdown is manual.

Supply these four JSONL files in a dataset directory:

```text
prompts_short_answers_short.jsonl
prompts_short_answers_long.jsonl
prompts_long_answers_short.jsonl
prompts_long_answers_long.jsonl
```

Each file must contain usable rows with `prompt`, `input_token_count` and
`output_token_count`. Counts are nonnegative integer reference lengths from the
target LLM tokenizer. File names select QPS streams; routing types are recomputed
using the configured input threshold and predicted output length.

## 3. Run the experiment

**Prepare phase 1.** Start its vLLM workers with the configured GPU/TP mapping,
model and ports. On each worker's host, start a matching Monitor; for example:

```bash
python monitor.py --worker-id phase1-ss --host 127.0.0.1 --port 8000 \
  --gpu-ids 0,1 --api-host 0.0.0.0 --api-port 9100 \
  --output-dir results/run01/phase1/monitor
```

Monitor stays idle until Router opens its recording window. Optionally start one
MIAD process per TP instance, using workload-specific thresholds and supported GPU
frequencies. These values are examples, not universal settings:

```bash
sudo /ABS/PATH/ENV/bin/python gpu_miad_controller.py \
  --gpu-ids 0,1 --vllm-ip 127.0.0.1 --vllm-port 8000 \
  --ttft-threshold-ms 600 --tbt-threshold-ms 150 --slo-threshold-ms 6300 \
  --min-frequency 300 --max-frequency 1410 --initial-frequency 1410 \
  --log-file results/run01/phase1/miad/worker.csv --reset-on-exit
```

**Start Router.** Set `INPUT_THRESHOLD` to your desired input boundary in tokens;
it is required rather than assigned an assumed default. Output threshold defaults
to 444. Use a fresh results directory for every run.

```bash
python router.py \
  --experiment-id run01 --dataset-directory /ABS/PATH/DATASET \
  --input-threshold "${INPUT_THRESHOLD:?Set INPUT_THRESHOLD in tokens}" \
  --output-threshold 444 \
  --phase1-config configs/phase1.json --phase2-config configs/phase2.json \
  --qps-config configs/qps.json --request-log results/run01/requests.jsonl \
  --duration 3600 --segment-seconds 300 --switch-at-seconds 1800 \
  --predictor-kwargs '{"device":"cpu"}'
```

Each QPS array needs `ceil(duration / segment-seconds)` entries. If omitted,
`--switch-at-seconds` defaults to half the duration. Other queue/concurrency options
are listed by `python router.py --help`.

**Start Controller** in another terminal:

```bash
python experiment_controller.py configs/controller.json
```

**At the pause**, stop phase-1 MIAD and the vLLM workers being replaced, and release
their GPUs. Retained vLLM workers may stay running. Update phase2.json if necessary.
Start every phase-2 Monitor with matching phase-2 IDs/GPU lists and a separate output
directory, plus phase-2 MIAD if desired. Leave new vLLM workers stopped. When GPUs,
SSH/service definitions and Monitors are ready, press Enter in Controller; it starts
new vLLM workers after the clock resumes. For noninteractive runs, configure a fresh
`ready_file` and create it at this point instead.

**After completion**, collect the three Router JSONL files and remote Monitor/MIAD
CSVs. Stop MIAD with Ctrl+C (`--reset-on-exit` resets clocks), remaining vLLM services
and Monitor processes. Router keeps its API alive for final status/cleanup; stop it
with Ctrl+C. JSONL logs overwrite same-name files; MIAD appends to same-name CSVs.

## Replaceable prediction and measurement scope

The supplied `TokenCountPredictor` in `Bert_Classifier/predict.py` uses XLM-RoBERTa
with a hierarchical LSTM regression head. `predict(prompt_text)` returns an estimated
output token count. Input/output values **greater than or equal to their respective
thresholds** are long. The paper's output boundary is 444 tokens; the project authors
report approximately **89% short/long response classification accuracy for
Llama3-70B on their evaluation dataset at that boundary**. This result is specific
to that model, dataset and threshold.

The predictor is a replaceable imported component. Any class/factory returning an
object with `predict(prompt_text) -> token_count` can be selected through:

```bash
# Add these options to the Router command; the module must be importable:
--predictor-factory my_predictor:MyPredictor --predictor-kwargs '{"model_dir":"/path/to/model"}'
```

The factory is called once before timing starts. For the built-in predictor,
`model_dir` overrides the weights/tokenizer directory and `device` selects `cpu`
(default) or a dedicated GPU such as `cuda:1`. A stronger compatible predictor can
improve routing accuracy; evaluate its inference overhead as well. Prediction
failures are logged, without falling back to ground-truth routing.

Prediction affects **routing only**. Generation limits remain the reference output
count clamped to 1..4096; the inherited Router SLO uses source-type TTFT/TBT constants
and reference output length, separately from MIAD thresholds. Logs separate reference
and predicted lengths. Router E2E includes prediction/queue waiting, inference,
startup and retries, but excludes manual freezes. Hard-cutoff discards appear in the
switch log rather than the main request log. Monitor records per-worker phase-1,
phase-2 startup and phase-2 running windows; MIAD may also record preparation time.

Run the regression tests (mock predictor/vLLM/GPU, no weights required):

```bash
python -m unittest discover -s tests -v
```

The HTTP test uses loopback sockets and skips when local binding is forbidden.
