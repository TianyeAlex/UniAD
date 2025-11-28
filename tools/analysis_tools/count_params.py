import argparse
import os
import sys

import torch
from mmcv import Config, DictAction

from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Count UniAD model parameters")
    parser.add_argument("config", help="Path to the model config file")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help=(
            "Override some settings in the config, key=val format. "
            "For list values use key=""[a,b]"" or key=a,b."
        ),
    )
    return parser.parse_args()


def import_custom_modules(cfg, config_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    if not hasattr(cfg, "plugin") or not cfg.plugin:
        return

    import importlib

    if hasattr(cfg, "plugin_dir"):
        module_dir = cfg.plugin_dir
    else:
        module_dir = os.path.dirname(config_path)

    module_dir = module_dir.rstrip("/")
    if module_dir.endswith(".py"):
        module_dir = os.path.dirname(module_dir)

    if not module_dir:
        return

    module_name = ".".join(part for part in module_dir.split("/") if part)
    if not module_name:
        return

    importlib.import_module(module_name)


@torch.no_grad()
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    import_custom_modules(cfg, args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    model.init_weights()

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(
        f"Trainable parameters: {trainable_params:,} "
        f"({trainable_params / 1e6:.2f} M)"
    )


if __name__ == "__main__":
    main()
