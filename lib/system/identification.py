import socket
import os
import torch
from datetime import datetime


def set_gpu_environment(cfg):
    # set the GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.SYSTEM.CUDA_VISIBLE_DEVICES
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device(cfg.SYSTEM.DEVISE)

    print("device is : {}".format(device))
    print('number of GPUs: {}'.format(torch.cuda.device_count()))
    if str(device) == 'cuda':
        print("current_device: {} device name: {}".format(torch.cuda.current_device(),
                                                          torch.cuda.get_device_name(torch.cuda.current_device())))
    return device


def merge_system_environment_to_cfg(cfg):
    root = "./config/system"
    host_name_dic = {'ARAS': 'ARAS.yml', 'DESKTOP-MT45091': 'W_sina.yml'}
    host_name = socket.gethostname()
    if host_name in host_name_dic:
        print("Host Name : {}".format(host_name))
        print('Work Space Name : {}'.format(host_name_dic[host_name]))
        cfg.merge_from_file(os.path.join(root, host_name_dic[host_name]))
    else:
        print("Host Name :Google_Colab -->{}".format(host_name))
        # print('Work Space Name : {}'.format(host_name_dic[host_name]))
        host_name = "Colab"
        # print('Work Space Name : {}'.format(host_name_dic[host_name]))
        cfg.merge_from_file(os.path.join(root, 'Colab.yml'))

    cfg.LOGGING.HOST = host_name
    cfg.LOGGING.TIME = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    weight_folder = os.path.join("./logs/", cfg.MODEL.NAME, cfg.TRAIN.STEP, host_name, cfg.LOGGING.COMMENT, cfg.LOGGING.TIME, 'weights/')
    if os.path.exists(weight_folder):
        print("weight folder exists!")
    else:
        os.makedirs(weight_folder)
        print("creat weight folder")
    cfg.LOGGING.WEIGHT_FOLDER = weight_folder

    log_dir = './logs/{}/{}/{}/{}/{}/'.format(cfg.MODEL.NAME, cfg.TRAIN.STEP, cfg.LOGGING.HOST, cfg.LOGGING.COMMENT, cfg.LOGGING.TIME)
    try:
        os.makedirs(log_dir)
        print("Directory ", log_dir, " Created ")
    except FileExistsError:
        print("Directory ", log_dir, " already exists")
    cfg.LOGGING.LOG_DIR = log_dir

    if cfg.TRAIN.BATCH_SIZE > cfg.SYSTEM.MAX_BATCH_SIZE:
        print("[warning] system configuration force batch size to: {}".format(cfg.SYSTEM.MAX_BATCH_SIZE))
        cfg.TRAIN.BATCH_SIZE = cfg.SYSTEM.MAX_BATCH_SIZE
    return cfg
