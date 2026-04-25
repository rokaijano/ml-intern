# GCP Experiment Tool

`run_gcp_experiment` is a narrow wrapper for running a bounded experiment on a
disposable Google Cloud VM from a pre-approved instance template. ml-intern does
not expose arbitrary `gcloud` access and does not run a shell command directly.
It only invokes this script inside the research repo:

```bash
python scripts/run_gcp_experiment.py \
  --branch <branch> \
  --template <template> \
  --zone <zone> \
  --minutes <minutes> \
  --command <command>
```

The external script is responsible for git push or checkout behavior, creating
the VM from the template, running the experiment, collecting logs and results,
destroying the VM, and writing run manifests.

## Configuration

The tool is registered when `enable_gcp_experiment_tool` is true in
`configs/main_agent_config.json`.

Environment variables:

- `ML_INTERN_GCP_ALLOWED_TEMPLATES`: comma-separated allowlist of GCP instance
  template names. Required for any run.
- `ML_INTERN_GCP_MAX_MINUTES`: maximum requested runtime in minutes. Defaults to
  `60`.
- `ML_INTERN_GCP_ALLOW_ANY_BRANCH`: set to `true` to allow branches that do not
  start with `exp/`. By default only `exp/*` branches are accepted.

## Example

```json
{
  "repo_path": "/path/to/research-repo",
  "branch": "exp/attention-ablation",
  "template": "ml-intern-a100-template",
  "zone": "us-central1-a",
  "minutes": 45,
  "command": "python experiments/run_attention_ablation.py --limit 1000"
}
```

## Safety Constraints

- The template must be in `ML_INTERN_GCP_ALLOWED_TEMPLATES`.
- `minutes` must be positive and no greater than `ML_INTERN_GCP_MAX_MINUTES`.
- Branches must start with `exp/` unless explicitly overridden.
- `repo_path` must exist and contain `scripts/run_gcp_experiment.py`.
- ml-intern invokes the wrapper with a subprocess argv list and `shell=False`.
- ml-intern does not call arbitrary `gcloud` commands.
- Tool output is structured JSON with `success`, `command_invoked`, `stdout`,
  `stderr`, and `exit_code`. Known token formats are scrubbed from output, but
  the external script should avoid printing secrets.
