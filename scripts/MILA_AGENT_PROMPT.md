# Prompt: run the Sentionaut GPU experiments on the Mila cluster

You are an agent running on Jacob's own machine, which has SSH access to
the Mila cluster. Your job is to run three GPU experiments from the
Sentionaut repository on Mila, check that they are sound, collect their
results, turn them into the documentation figures and tables, and push those
to the branch. You submit jobs; you never run compute on a login node.

## 0. Facts you need

- Repository: `git@github.com:ascientist/sentionaut.git`, branch
  `claude/zen-cori-aun5ja`. Everything below refers to files on that branch.
- Cluster access: `ssh mila` (Mila's standard alias, set up by `mila init`;
  it reaches `login.server.mila.quebec` on port 2222). If the alias is
  missing, stop and ask Jacob. Do not try to configure SSH keys yourself.
- On the cluster the checkout lives in `$HOME/sentionaut`. Results, logs and
  caches go to `/network/scratch/j/jacob.lavoie/sentionaut` (`$SCRATCH/sentionaut`).
- Mila rules you must respect (https://docs.mila.quebec):
  - **No compute on login nodes.** Only `git`, `sbatch`, `squeue`, `sacct`,
    `scancel`, `ls`, `cat`, `tail` and file copies there. `uv sync` and any
    Python that imports torch run inside jobs (`scripts/mila_prepare.sh` does
    it on the GPU node).
  - Partition **`main`**: at most 2 GPUs, 8 CPUs and 48 GB per job, 5 days.
    `main` jobs are not preempted by other `main` jobs. Do not move the timed
    runs to `long` (preemptible; a requeued job restarts from scratch and its
    wall-clock is meaningless) or to `unkillable` (6 CPUs, 32 GB, 1 GPU; for
    interactive work).
  - GPU type: **L40S**, requested as `--gpus-per-task=l40s:1` (91 nodes × 4
    GPUs, 48 GB each; per-GPU bundle 12 CPUs, 250 GB). Every timed task must
    use the same GPU type, so do not change it to "any GPU".
  - Use `$SLURM_TMPDIR` for job-local outputs and copy to scratch at the end
    (the scripts already do this). Scratch is not backed up.
  - Keep at most 8 array tasks running at once (`%8` in the script). If
    `squeue` shows `QOSMaxJobsPerUserLimit`, that is the cluster's per-user
    limit: wait, do not resubmit.

## 1. What the three jobs do

| Script | What | Resources | Expected time |
| --- | --- | --- | --- |
| `scripts/mila_imagine.sh` | Trains a vision encoder (image → electrode amplitudes) four ways on two simulators, three seeds: 24 array tasks. Arms: backprop through the analytical teacher, backprop through a world model fitted online (the SuperMimic loop), REINFORCE in the teacher, REINFORCE in the world model. Teachers: retinal axon map (Argus II), cortical Dynaphos (Orion). Each task trains for a 30-minute wall-clock budget, then evaluates on the teacher. | `main`, 1 × L40S, 4 CPUs, 24 GB, 1 h 15 min per task | ~3 h for all 24 at 8 concurrent |
| `scripts/mila_bench_forward.sh` | Throughput (samples/s) of axon-map forward models against batch size: pulse2percept's cython engine on CPU (1 and 8 threads) and its jax engine on GPU, which are third-party; the repo's torch port (exact and 5σ-sparse) and its learned world model on GPU, which are ours. Forward and forward+backward. | `main`, 1 × L40S, 8 CPUs, 32 GB, 1 h 30 min | < 1 h |
| `scripts/mila_fit_drawings.sh` | Fits the axon map's `rho`, `axlambda` and a drawing threshold to the phosphene drawings of three Argus II users (Beyeler et al. 2019, downloaded from OSF with a SHA-256 check), by gradient descent, tested on held-out electrodes. | `main`, 1 × L40S, 4 CPUs, 32 GB, 2 h | < 1 h |

Every job first sources `scripts/mila_prepare.sh`. Under a file lock, it runs
`uv sync --locked`, checks that CUDA is visible, and builds the MNIST and
axon-map caches on scratch, so concurrent tasks do not race on them.

## 2. Procedure

### 2.1 Update the checkout (login node)

```bash
ssh mila
cd $HOME/sentionaut
git status --short            # must be clean; if not, stop and report what is dirty
git fetch origin
git checkout claude/zen-cori-aun5ja
git pull --ff-only origin claude/zen-cori-aun5ja
git log --oneline -3          # report the commit you are running
mkdir -p /network/scratch/j/jacob.lavoie/sentionaut/logs   # Slurm writes logs here; it must exist
```

If `$HOME/sentionaut` does not exist, clone it there first. If uv is not on
`PATH` (`ls ~/.local/bin/uv`), stop and ask; do not install it from a job.

### 2.2 Smoke test first (one task, 60-second budget)

Never start the full array before one short task has gone end to end.

```bash
BUDGET_S=60 sbatch --array=0 --time=00:30:00 scripts/mila_imagine.sh
BUDGET_S=60 sbatch --array=15 --time=00:30:00 scripts/mila_imagine.sh   # dynaphos, wm-bptt, seed 0
squeue --me
```

`sbatch` propagates the environment, so `BUDGET_S` reaches the job. Task 0
is axon map, backprop through teacher, seed 0 (it also computes the axon-map
baselines). Task 15 is Dynaphos, world-model arm, seed 0.

When both are done (`sacct -j <id> --format=JobID,State,ExitCode,Elapsed,MaxRSS`
shows `COMPLETED`), check:

```bash
tail -n 30 /network/scratch/j/jacob.lavoie/sentionaut/logs/sentionaut-imagine_<jobid>_0.out
ls /network/scratch/j/jacob.lavoie/sentionaut/imagine/runs/
```

You should see a line `gpu NVIDIA L40S`, a final JSON line with `t`,
`teacher_steps` and `loss`, and the files `axonmap_teacher-bptt_s0.json`,
`axonmap_baselines.json` and `dynaphos_wm-bptt_s0.json` (plus `_encoder.pt`
files). Open one run JSON and check that `"device"` is `"NVIDIA L40S"` and
that `log` has more than two entries.

Delete the smoke results so they do not mix with the real runs:

```bash
rm /network/scratch/j/jacob.lavoie/sentionaut/imagine/runs/*
```

### 2.3 Full runs

```bash
sbatch scripts/mila_imagine.sh          # 24 tasks
sbatch scripts/mila_bench_forward.sh
sbatch scripts/mila_fit_drawings.sh
```

Monitor with `squeue --me` and `sacct -u $USER --start=today
--format=JobID%20,JobName,State,ExitCode,Elapsed,MaxRSS`. Do not poll faster
than every few minutes. Pending reasons `(Priority)` and `(Resources)` are
normal.

### 2.4 When a job fails

Read the job's log in `/network/scratch/j/jacob.lavoie/sentionaut/logs/` and
`sacct` first. Then:

| Symptom | Action |
| --- | --- |
| `OUT_OF_MEMORY` (host RAM) | Resubmit only that task with more memory, within the 48 GB cap: `sbatch --array=<id> --mem-per-gpu=40G scripts/mila_imagine.sh`. |
| `CUDA out of memory` in the log | Report it with the task id and arm; do not change batch sizes yourself (that would make arms incomparable). |
| `TIMEOUT` | Resubmit that task once with `--time=02:00:00`. If it times out again, report it. |
| `CUDA required` assertion, or no `gpu NVIDIA L40S` line | Node or environment problem. Resubmit once; if it repeats, report it. |
| `uv sync --locked` fails | The lockfile does not match `pyproject.toml`. Stop and report; do not run `uv lock` on the cluster. |
| A Python traceback in our code | Stop that experiment, report the traceback and the commit. Do not patch experiment code on the cluster. |
| Stuck on the lock (`prepare.lock`) for > 1 h | Check for a dead job holding it (`squeue --me`); report rather than deleting the lock while jobs run. |

In the benchmark, `"error"` lines saying out of memory at the largest batch
of `torch-exact` forward+backward are expected. Keep them, they are a result.

### 2.5 Collect and build the report (back on your machine)

```bash
cd <your local clone of sentionaut>
git checkout claude/zen-cori-aun5ja && git pull --ff-only
mkdir -p data/mila
rsync -av mila:/network/scratch/j/jacob.lavoie/sentionaut/imagine/runs/ data/mila/imagine/
rsync -av mila:/network/scratch/j/jacob.lavoie/sentionaut/bench/ data/mila/bench/
rsync -av mila:/network/scratch/j/jacob.lavoie/sentionaut/p2p/drawing_fits.json data/mila/
uv run python scripts/imagine_report.py --runs data/mila/imagine --out docs/assets/imagination
```

`data/` is gitignored; only the report output in `docs/assets/imagination/`
(PNGs and `results.md`) and the small JSON summaries you copy into
`docs/assets/` are committed. Then:

1. Copy the run JSONs (not the `.pt` encoders) to `docs/assets/imagination/runs/`,
   the benchmark JSONL files to `docs/assets/bench/`, and `drawing_fits.json`
   to `docs/assets/patient-data/`.
2. In `docs/imagination.md`, add a "GPU results (Mila, L40S)" section that
   includes the two `*_curves.png` figures and the tables from `results.md`.
   Say which arm reaches the phosphene-map baseline first in wall-clock and in
   teacher steps, per teacher, and whether that holds for all three seeds.
3. In `docs/patient-data.md`, replace "Fit results will be added once the fits
   have run." with a table per subject: fitted `rho`, `axlambda`, threshold,
   test loss, and default-parameter test loss with its terms.
4. Add a "Throughput" section to `docs/imagination.md` with a table of
   samples/s by engine and batch (forward and forward+backward), naming the
   pulse2percept engines as third-party.
5. Run `uv run ruff check src scripts tests` and
   `uv run pytest -q -m "not slow"`, and commit with a message that says what
   was run, on which commit, on which GPU. Push to `claude/zen-cori-aun5ja`
   only. Do not open a pull request.

### 2.6 What to report back to Jacob

- The commit you ran, the job ids, and anything resubmitted or failed, with why.
- Per teacher: the final loss of each arm (mean and range over seeds) and the
  phosphene-map and zero-stimulation baselines.
- Whether the world-model arm still collapses to no stimulation (charge near
  0). That failure was seen in the CPU pilot before the exploration fix; say
  plainly if it recurs.
- The benchmark table, highlighting the batch size at which each engine stops
  scaling.
- The drawing fits: do fitted parameters beat the defaults on held-out
  electrodes, for each subject?
- Totals: GPU-hours used (`sacct` elapsed × GPUs).

Do not interpret beyond the numbers. If a result looks wrong (a loss of
exactly 0, `nan`, an arm with no evaluations), say so and do not polish it.
