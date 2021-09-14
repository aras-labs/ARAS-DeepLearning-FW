import argparse
from torch.utils.tensorboard import SummaryWriter
from tensorboard.summary import _writer
from lib.system.identification import merge_system_environment_to_cfg, set_gpu_environment


def init_env(cfg, has_writer=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file path")
    args = parser.parse_args()
    print(args.config)

    cfg.merge_from_file(args.config)
    cfg = merge_system_environment_to_cfg(cfg)
    cfg.freeze()

    print(cfg.dump())

    with open(cfg.LOGGING.LOG_DIR + 'config.yml', 'w') as f:
        f.write(cfg.dump())

    device = set_gpu_environment(cfg)
    if has_writer:
        # writer = SummaryWriter(cfg.LOGGING.LOG_DIR)
        writer = _writer.Writer(cfg.LOGGING.LOG_DIR)

        return device, writer
    return device
