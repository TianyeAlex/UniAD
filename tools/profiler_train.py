from __future__ import division

import argparse
import copy
import os
import time
import warnings

import cv2
import mmcv
import sklearn
import torch
from mmcv import Config, DictAction
from mmcv.runner import HOOKS, Hook, get_dist_info, init_dist
from os import path as osp

from torch.profiler import (profile, schedule,
                            tensorboard_trace_handler)

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

warnings.filterwarnings("ignore")

from mmcv.utils import TORCH_VERSION, digit_version


class TorchProfilerHook(Hook):
    def __init__(
        self,
        log_dir,
        wait=2,
        warmup=3,
        active=10,
        repeat=1,
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        profile_memory=False,
        export_mode="chrome",
    ):
        super().__init__()
        self.log_dir = log_dir
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        valid_modes = {"tensorboard", "chrome", "none"}
        if export_mode not in valid_modes:
            raise ValueError(
                f"export_mode must be one of {sorted(valid_modes)}, got {export_mode}"
            )
        self.export_mode = export_mode
        self.profiler = None
        self._trace_idx = 0

    def before_run(self, runner):
        mmcv.mkdir_or_exist(osp.abspath(self.log_dir))
        sched = schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat,
        )

        on_trace_ready = None

        if self.export_mode == "chrome":
            def _save_trace(prof):
                trace_path = osp.join(
                    self.log_dir, f"trace_{self._trace_idx:03d}.json")
                prof.export_chrome_trace(trace_path)
                self._trace_idx += 1
            on_trace_ready = _save_trace
        elif self.export_mode == "tensorboard":
            try:
                # TensorBoard profiler plugin expects: logdir/plugins/profile/YYYY_MM_DD_HH_MM_SS/
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                tb_profile_dir = osp.join(self.log_dir, "plugins", "profile", timestamp)
                mmcv.mkdir_or_exist(tb_profile_dir)
                
                tb_handler = tensorboard_trace_handler(tb_profile_dir)
                print(f"[TorchProfilerHook] TensorBoard handler initialized → {tb_profile_dir}")
                
                def _wrapped_handler(prof):
                    print(f"[TorchProfilerHook] Writing TensorBoard trace...")
                    try:
                        tb_handler(prof)
                        print(f"[TorchProfilerHook] TensorBoard trace written successfully to {tb_profile_dir}")
                    except Exception as e:
                        print(f"[TorchProfilerHook] TensorBoard write failed: {e}")
                        raise
                
                on_trace_ready = _wrapped_handler
            except Exception as e:
                print(f"[TorchProfilerHook] TensorBoard handler init failed: {e}, falling back to chrome trace")
                def _save_trace(prof):
                    trace_path = osp.join(
                        self.log_dir, f"trace_{self._trace_idx:03d}.json")
                    prof.export_chrome_trace(trace_path)
                    self._trace_idx += 1
                on_trace_ready = _save_trace
        # else: on_trace_ready stays None

        self.profiler = profile(
            schedule=sched,
            on_trace_ready=on_trace_ready,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            profile_memory=self.profile_memory,
        )
        self.profiler.start()

    def after_train_iter(self, runner):
        if self.profiler is not None:
            self.profiler.step()

    def after_run(self, runner):
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None


if 'TorchProfilerHook' not in HOOKS.module_dict:
    HOOKS.register_module(module=TorchProfilerHook)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed') 
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument(
        '--profiler-disable',
        action='store_true',
        help='disable torch profiler hook')
    parser.add_argument(
        '--profiler-dir',
        default=None,
        help='output directory for profiler traces (default: work_dir/profiler_logs)')
    parser.add_argument('--profiler-wait', type=int, default=10, help='profiler wait steps before warmup')
    parser.add_argument('--profiler-warmup', type=int, default=1, help='profiler warmup steps')
    parser.add_argument('--profiler-active', type=int, default=3, help='profiler active steps to record')
    parser.add_argument('--profiler-repeat', type=int, default=1, help='profiler schedule repeat count')
    parser.add_argument(
        '--profiler-export',
        choices=['tensorboard', 'chrome', 'none'],
        default='tensorboard',
        help='select profiler output target (default: tensorboard)')
    parser.add_argument(
        '--profiler-no-shapes',
        action='store_true',
        help='do not record operation input shapes in profiler')
    parser.add_argument(
        '--profiler-no-stack',
        action='store_true',
        help='disable stack trace capture in profiler')
    parser.add_argument(
        '--profiler-no-flops',
        action='store_true',
        help='disable FLOPs estimation in profiler')
    parser.add_argument(
        '--profiler-profile-memory',
        action='store_true',
        help='enable CUDA memory profiling')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            from projects.mmdet3d_plugin.uniad.apis.train import custom_train_model
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    if not args.profiler_disable:
        profiler_dir = args.profiler_dir or osp.join(cfg.work_dir, 'profiler_logs')
        cfg.custom_hooks = cfg.get('custom_hooks', [])
        cfg.custom_hooks.append(
            dict(
                type='TorchProfilerHook',
                log_dir=profiler_dir,
                wait=args.profiler_wait,
                warmup=args.profiler_warmup,
                active=args.profiler_active,
                repeat=args.profiler_repeat,
                record_shapes=not args.profiler_no_shapes,
                with_stack=not args.profiler_no_stack,
                with_flops=not args.profiler_no_flops,
                profile_memory=args.profiler_profile_memory,
                export_mode=args.profiler_export,
                priority='VERY_HIGH',
            )
        )

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # NOTE: We have adopted a more elegant way to set the logger_name
    logger_name = cfg.get('logger_name', 'mmdet')
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    # NOTE: To fix the serialization issue in nuScenes-dev-kit, we adopt this method to skip the pickle steps
    torch.multiprocessing.set_start_method('fork')
    main()
