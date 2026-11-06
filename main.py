"""Top-level pipeline execution entrypoint."""

import argparse
from src.utils.config import load_config
from src.utils.reproducibility import set_seed, get_device


def main():
    parser = argparse.ArgumentParser(description="Resource-Adaptive Gated NIDS Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--mode", type=str, choices=["audit", "preprocess", "train", "federated", "eval"], default="audit")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("project", {}).get("seed", 42)
    device_pref = config.get("project", {}).get("device", "auto")
    
    set_seed(seed)
    device = get_device(device_pref)

    print(f"==================================================")
    print(f"Resource-Adaptive Gated NIDS")
    print(f"Mode: {args.mode}")
    print(f"Resolved Device: {device}")
    print(f"==================================================")
    print("Project skeleton successfully initialized. Ready for module implementation.")


if __name__ == "__main__":
    main()

