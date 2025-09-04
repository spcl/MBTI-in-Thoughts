# Primed Multi-Agent Collaboration on Solving Benchmark Tasks

This directory contains the code for solving tasks from different benchmarks (BIG-Bench, SOCKET) using multiple differently primed agents and different collaboration strategies.
The main script allows you to specify the solver, dataset, dataset configuration, model, and agent types.

## Usage

You can adapt [run_big_bench_SOCKET.sh](run_big_bench_SOCKET.sh) to your needs before you run the script.

```bash
./run_big_bench_SOCKET.sh
```

### Arguments

- `--solver`: Method to solve the task. Choices are dynamically generated from files start with `solve_` and end with `.py`.
- `--dataset`: Task dataset. Choices are `Blablablab/SOCKET` and `test`.
- `--dataset_config`: A configuration of the Dataset SOCKET. Choices are dynamically generated from the `get_configs` function plus `test`.
- `--model`: Model name. Example: `gpt-4o-mini`.
- `--agent_types`: List of agent types. Example: `INTJ ENFP ISTP`.

### Possible tasks from the datasets

- Tasks that you can run are listed in [data/task_descriptions](data/task_descriptions).
