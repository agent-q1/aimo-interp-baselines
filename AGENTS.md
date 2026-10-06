# Repository guidance

- Use uv for Python environments and dependencies. Prefer `uv sync`, `uv run`,
  `uv add`, and uv-managed dependency groups when a project manifest exists.
  This repo currently uses `requirements.txt`: create `.venv` with `uv venv
  --python 3.12`, then use `uv pip install --python .venv/bin/python -r
  requirements.txt`. Avoid direct pip and ad hoc virtualenv workflows.
- Read `handover.txt` before running the GPU baseline. It records setup,
  commands, expected outputs, and known issues to verify.
- Focus on the first baseline: frozen-model activation extraction in
  `scripts/encode.py`, followed by linear probing (`--method probe`) in
  `scripts/probe.py`. Do not run the kernel baseline unless requested.
- Use `data/math-robust-final.csv` for the baseline. It has 675 rows and 137
  unique original prompts, including separate perturbation categories and
  `OVERALL`. Probes train separately per category and layer; there is no
  inference ensemble in this repo.
- Keep model weights, activation arrays, results, environments, and the cloned
  `holmes-evaluation` dependency out of Git. Keep the exploration notebook
  tracked; do not include credentials or tokens in notebooks or logs.
- Start with a small real-data smoke run and inspect metrics before the full
  sweep. Benchmark before estimating runtime. Use one probe worker on a single
  GPU because each worker independently selects the available GPU.
- Preserve the distinction between the model producing behavioral labels
  (`qwen3-8b:low` in the CSV) and the extraction model (the wrapper defaults to
  `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`). Record both in experiment notes.
- Branch: `sshankar/run-baseline`. The user's fork is
  `https://github.com/agent-q1/aimo-interp-baselines`; the original upstream is
  `https://github.com/aimo-interp/baselines`.
