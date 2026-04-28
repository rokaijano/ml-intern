# GCP Experiment Tool

`run_gcp_experiment` is a narrow wrapper for running a bounded experiment through
the research repo's own dispatcher. The repo decides from its environment
whether the run is local, on a persistent GCP VM, or on a disposable GCP VM.
ml-intern does not expose arbitrary `gcloud` access and does not run a shell
command directly. It only invokes this script inside the research repo:

```bash
python scripts/run_gcp_experiment.py \
  --branch <branch> \
  --minutes <minutes> \
  --command <command> \
  [--mode auto|local|persistent|disposable]
```

The external script is responsible for git handoff, mode selection, VM lifecycle,
running the experiment, collecting logs and results, destroying disposable VMs,
and writing run manifests.

## Configuration

The tool is registered when `enable_gcp_experiment_tool` is true in
`configs/main_agent_config.json`.

Environment variables:

- `ML_INTERN_GCP_ALLOWED_TEMPLATES`: comma-separated allowlist of GCP instance
  template names. Used by the research repo for disposable mode.
- `ML_INTERN_GCP_ALLOWED_INSTANCES`: optional comma-separated allowlist of
  persistent VM names. Used by the research repo for persistent mode.
- `ML_INTERN_GCP_MAX_MINUTES`: maximum requested runtime in minutes. Defaults to
  `60`.
- `ML_INTERN_GCP_ALLOW_ANY_BRANCH`: set to `true` to allow branches that do not
  start with `exp/`. By default only `exp/*` branches are accepted.
- `ML_INTERN_GCP_SINGLE_FLIGHT`: default `true`; ml-intern refuses to launch a
  second experiment while one is already running.

## Example

```json
{
  "repo_path": "/path/to/research-repo",
  "branch": "exp/attention-ablation",
  "minutes": 45,
  "command": "python experiments/run_attention_ablation.py --limit 1000",
  "mode": "auto"
}
```

## Safety Constraints

- `minutes` must be positive and no greater than `ML_INTERN_GCP_MAX_MINUTES`.
- Branches must start with `exp/` unless explicitly overridden.
- `repo_path` must exist and contain `scripts/run_gcp_experiment.py`.
- Single-flight locking is enabled by default.
- ml-intern invokes the wrapper with a subprocess argv list and `shell=False`.
- ml-intern does not call arbitrary `gcloud` commands.
- Tool output is structured JSON with `success`, `command_invoked`, `stdout`,
  `stderr`, and `exit_code`. Known token formats are scrubbed from output, but
  the external script should avoid printing secrets.
