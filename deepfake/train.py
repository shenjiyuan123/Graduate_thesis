import os
import sys
import argparse
import math
import random
import datetime 
import numpy as np
from prometheus_client import Summary
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from mmcv import Config
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
from deepfake.datasets import Deepfake_datasets, build_dataloader
from deepfake.utilis import init_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('cfg', help='configuration')
    args = parser.parse_args()
    return args

class Logger(object):
    def __init__(self, filename = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w+", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def set_fixed_seed(cfg_seed):
    seed = cfg_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train():
    # init
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    writer = SummaryWriter()
    # Some trivial things
    # make checkpoint folder
    if not os.path.exists(cfg.model_save_path):
        os.makedirs(cfg.model_save_path)
    # save the logger
    sys.stdout = Logger(cfg.log_file)
    best_model   = os.path.join(cfg.model_save_path, 'best_model.pth')
    latest_model = os.path.join(cfg.model_save_path, 'latest_model.pth')
    # set the fixed seed
    set_fixed_seed(cfg.seed)
    # define cuda
    print(f"Cuda available devices: {torch.cuda.device_count()}.")
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = init_model(cfg, mode='train', type=cfg.model_type, path=cfg.pretrained_path, writer=writer)
    if cfg.multi_gpu:
        model = nn.DataParallel(model).to(device)
    else:
        model.to(device)
    # build the dataset
    # transform setting
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # train datasets
    train_dataset    = Deepfake_datasets(cfg.traindata, transform=transform, use_DCT=cfg.use_DCT)
    train_step_len   = math.ceil((train_dataset.__len__())/(cfg.batch_size))
    train_dataloader = build_dataloader(train_dataset,batch_size=cfg.batch_size,num_workers=cfg.num_workers)
    # eval datasets
    eval_dataset    = Deepfake_datasets(cfg.evaldata, transform=transform, use_DCT=cfg.use_DCT)
    eval_dataloader = build_dataloader(eval_dataset, batch_size=cfg.batch_size,num_workers=cfg.num_workers)
    # setting configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p:p.requires_grad, model.parameters()), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay,
        betas=cfg.betas
        )
    # train
    best_f1 = 0.0
    for epoch in range(cfg.epoch_num):
        model.train()
        acc, f1, running_loss = 0.0, 0.0, 0.0
        for i, data in enumerate(train_dataloader):
            input, label = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            output = model(input)
            if type(output) is tuple:
                stu_fea, stu_cla = output[0], output[1]
            elif type(output) is torch.Tensor:
                stu_cla = output
            loss = criterion(stu_cla, label)
            loss.backward()
            optimizer.step()
            # compute acc, f1, running_loss
            predict = stu_cla.data.cpu().numpy().argsort()[:,-1:].squeeze()
            acc += accuracy_score(label.cpu(),predict)
            f1  += f1_score(label.cpu(),predict,average='macro')
            running_loss += loss
            # for every interval, print log. i begins from 0 to 19 equals to 20
            if (i+1) % cfg.log_interval == 0:
                step = cfg.log_interval
                interval_acc  = acc/step
                interval_f1   = f1/step
                interval_loss = running_loss
                writer.add_scalar("Loss/train", running_loss, train_step_len*epoch+i+1)
                writer.add_scalar("Acc/train", interval_acc, train_step_len*epoch+i+1)
                writer.add_scalar("F1/train", interval_f1, train_step_len*epoch+i+1)
                print(f"{datetime.datetime.now()} -Train- Epoch [{epoch+1}][{i+1}/{train_step_len}]: loss:{interval_loss}, acc:{interval_acc}, f1_score:{interval_f1}")
                acc, f1, running_loss = 0.0, 0.0, 0.0
        # eval when training
        if (epoch+1) % cfg.evaluation_interval == 0:
            print("--------------------------------------------------------------------------")
            print(f"Evaluation after [{epoch+1}/{cfg.epoch_num}] epochs...")
            model.eval()
            acc, f1, eval_loss = 0.0, 0.0, 0.0
            with torch.no_grad():
                for j, data in enumerate(eval_dataloader):
                    input, label = data[0].to(device), data[1].to(device)
                    output = model(input)
                    if type(output) is tuple:
                        stu_fea, stu_cla = output[0], output[1]
                    elif type(output) is torch.Tensor:
                        stu_cla = output
                    loss = criterion(stu_cla, label)
                    predict = stu_cla.data.cpu().numpy().argsort()[:,-1:].squeeze()
                    acc += accuracy_score(label.cpu(),predict)
                    f1  += f1_score(label.cpu(),predict,average='macro')
                    eval_loss += loss
            # get the metrics
            total_batch_step = j + 1
            eval_accuracy = acc/total_batch_step
            eval_f1       = f1/total_batch_step
            writer.add_scalar("F1/eval", eval_f1, epoch+1)
            print(f"{datetime.datetime.now()} -Eval- Loss:{eval_loss}, acc:{eval_accuracy}, f1_score:{eval_f1}")
            # save the best model
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                torch.save(model.state_dict(), best_model)
                print(f"Now best checkpoint is {best_f1}, saved as {best_model}")
            print("--------------------------------------------------------------------------")
    writer.flush()
    torch.save(model.state_dict(), latest_model)
    print("Finish train.~")



def main():
    train()

if __name__=="__main__":
    main()

'''
command take-away note:
CUDA_VISIBLE_DEVICES=1,2,3 python train.py configs/f3net.py 
'''