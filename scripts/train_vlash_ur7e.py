"""Prepare/train the UR7E OpenPI VLASh recipe without touching other runs."""

import argparse
import dataclasses
import json
import os
import pathlib
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--init-checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--exp-name", default="gui_run")
    parser.add_argument("--checkpoint-base-dir", default="/data/qmc/openpi_data/checkpoints")
    parser.add_argument("--num-train-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--max-delay-steps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--stage", choices=("all", "preflight", "norm", "train"), default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_config(args):
    from openpi.training import config
    from openpi.training.vlash_config import VlashWeightLoader

    if args.max_delay_steps < 1 or args.batch_size < 1 or args.num_train_steps < 1 or args.max_frames < args.batch_size:
        raise ValueError("Invalid VLASh delay/batch/training/norm-stat settings")
    if args.action_horizon != 50:
        raise ValueError("The deployed UR7E recipe uses action_horizon=50; n_action_steps may be shorter")
    if not args.exp_name or pathlib.Path(args.exp_name).name != args.exp_name or args.exp_name in (".", ".."):
        raise ValueError("exp_name must be one directory name")
    dataset_root = args.dataset_root.resolve()
    info = json.loads((dataset_root / "meta/info.json").read_text())
    features = info["features"]
    for name in ("observation.state", "action"):
        if features[name]["shape"] != [7]:
            raise ValueError("UR7E VLASh requires seven absolute joint/gripper dimensions")
    for name in ("observation.images.camera_0", "observation.images.camera_1"):
        if name not in features:
            raise ValueError(f"Dataset is missing {name}")
    if float(info["fps"]) != 30:
        raise ValueError("This deployment expects a 30 Hz dataset")
    if not (args.init_checkpoint / "params").is_dir():
        raise ValueError("init-checkpoint must point to an OpenPI step directory containing params")
    # The pinned OpenPI LeRobot loader resolves repo ids relative to HF_LEROBOT_HOME.
    os.environ["HF_LEROBOT_HOME"] = str(dataset_root.parent)
    base = config.get_config("pi05_ur7e_vlash_lora")
    return dataclasses.replace(
        base, exp_name=args.exp_name,
        data=dataclasses.replace(base.data, repo_id=dataset_root.name,
            base_config=dataclasses.replace(base.data.base_config, vlash_max_delay_steps=args.max_delay_steps)),
        weight_loader=VlashWeightLoader(str(args.init_checkpoint.resolve() / "params")),
        checkpoint_base_dir=args.checkpoint_base_dir, num_train_steps=args.num_train_steps,
        batch_size=args.batch_size, num_workers=args.num_workers,
        save_interval=args.save_interval, log_interval=args.log_interval, resume=args.resume,
    )


def main():
    args = parse_args()
    if args.stage == "all":
        for stage in ("norm", "train"):
            env = dict(os.environ)
            if stage == "norm":
                env["JAX_PLATFORMS"] = "cpu"
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--stage", stage], env=env, check=True)
        return
    # Set the dataset root before importing LeRobot (it caches its home at import).
    os.environ["HF_LEROBOT_HOME"] = str(args.dataset_root.resolve().parent)
    cfg = make_config(args)
    if cfg.checkpoint_dir.exists() and not args.resume and args.stage in ("norm", "train"):
        raise FileExistsError(f"Training output already exists: {cfg.checkpoint_dir}; choose another exp-name or --resume")
    print(f"VLASh: output={cfg.checkpoint_dir}; delay=[0,{args.max_delay_steps}]; state=future_absolute_action", flush=True)
    if args.stage == "preflight":
        from openpi.training import data_loader

        data = cfg.data.create(cfg.assets_dirs, cfg.model)
        dataset = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model)
        sample = dataset[0]
        print(f"Preflight OK: frames={len(dataset)}, action_shape={sample['action'].shape}, valid={sample['action_loss_mask'].sum()}")
    elif args.stage == "norm":
        from openpi.training import config
        import compute_norm_stats

        config._CONFIGS_DICT[cfg.name] = cfg
        compute_norm_stats.main(cfg.name, max_frames=args.max_frames)
    else:
        import train

        train.main(cfg)


if __name__ == "__main__":
    main()
