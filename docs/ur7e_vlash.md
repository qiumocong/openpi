# UR7E VLASh deployment

Source reference: https://github.com/mit-han-lab/vlash/tree/22cbabfee0f57874987c75a35a7dac129e695db0

This is an OpenPI JAX port of VLASh future-state-aware async training and
execution, not the upstream PyTorch runtime. It requires new fine-tuning.
It does not implement upstream fused attention, shared-observation training,
or action quantization; no upstream throughput claim applies to this port.

## Training contract

- Keep images and task at timestep t.
- Sample d uniformly in [0, max_delay_steps].
- For d=0 keep measured state; otherwise use absolute action a[t+d-1].
- Predict H actions starting at t+d. Episode-end padding is masked in loss.
- Feed normalized future state through a trainable state MLP into pi05 adaRMS.
- Normalize joint deltas relative to that same future state; gripper stays absolute.

Existing task weights initialize LoRA fine-tuning and the new state MLP.
New checkpoints include assets/vlash.json; ordinary checkpoints are rejected.

## GUI

Run vla_gui.pyw from wrs-qiu-vlash/0000_VLA using its shared main environment.
Select Pi05, remote training, joint positions, no offline smoothing and no
LeRobot evaluation. Select dataset0924, a new output folder name, and the
ordinary OpenPI step checkpoint as initialization. Batch/steps/chunk size
come from the GUI. OpenPI VLASh uses PyAV and H=50; LeRobot precision/eval
settings do not configure this JAX recipe.

The dataset folder name maps beneath remote.openpi_dataset_root's parent.
Output is remote.openpi_checkpoint_base/pi05_ur7e_vlash_lora/<output-folder-name>.
Training computes its own normalization stats before starting. Existing
training output is not overwritten. Resume is available through the CLI.
Do not launch training while another GPU service occupies the needed memory.

After training, select the matching Linux checkpoint run/step in the testing
tab. Port 8766 isolates this service from RTC on 8765. The GUI launches code
from /home/qmc/CodeProjects/openpi-vlash-worktree with the main OpenPI Python
environment and an explicit worktree PYTHONPATH.

## Execution

Prefetch uses current images plus the last absolute action of the executing
chunk as future state. The next chunk starts at that future boundary and its
head is not discarded. Prefetch must be within the trained delay range.
chunk_blend_steps must remain 0: early blending would invalidate the endpoint
used as the future-state anchor. Servo trajectory interpolation remains on.
This is no guarantee of physical tracking: start with no-motion validation,
then a supervised reduced-speed test using the existing emergency stop.

## Remote CLI

```bash
cd /home/qmc/CodeProjects/openpi-vlash-worktree
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/qmc/CodeProjects/openpi/.venv/bin/python scripts/train_vlash_ur7e.py \
  --dataset-root /home/qmc/CodeProjects/openpi/dataset0924 \
  --init-checkpoint /data/qmc/openpi_data/checkpoints/pi05_ur7e_base_lora/dataset0924_50stp/29999 \
  --exp-name gui_run --max-delay-steps 10 --batch-size 4 --num-train-steps 30000
```

Use --stage preflight to inspect the dataset without training, --stage norm
for statistics only, --stage train to use previously computed statistics,
and --resume to explicitly resume an existing run.

No full training or real robot motion was performed during installation.

## Background Launcher

In the remote VLASh worktree, run:

```bash
./run_pi05_training_background.sh dataset0924_vlash fresh --dry-run
./run_pi05_training_background.sh dataset0924_vlash fresh
tail -f logs/pi05_ur7e_vlash_lora_dataset0924_vlash.log
```

Fresh runs compute their normalization statistics on CPU before GPU training.
For an existing saved run, use `./run_pi05_training_background.sh dataset0924_vlash resume`;
this preserves the original normalization statistics. The launcher rejects
overwrite and reports the PID, log and checkpoint paths.

Override settings using environment variables, for example:

```bash
BATCH_SIZE=2 MAX_DELAY_STEPS=10 NUM_TRAIN_STEPS=30000 \
  ./run_pi05_training_background.sh dataset0924_vlash_bs2 fresh
```

Other overrides: DATASET_ROOT, INIT_CHECKPOINT, CHECKPOINT_ROOT, SAVE_INTERVAL,
LOG_INTERVAL, NUM_WORKERS, NORM_STATS_MAX_FRAMES, OPENPI_PYTHON.
Stop with the printed process-group command, not just its child training PID.
