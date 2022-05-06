import os
import json
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from deepfake.datasets import Deepfake_datasets, build_dataloader
from deepfake.utilis import init_model
from mmcv import Config
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Eval script')
    parser.add_argument('cfg', help='configuration')
    parser.add_argument('--out', help='output json path')
    args = parser.parse_args()
    return args

def eval():
    # init
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    model_path = os.path.join(cfg.model_save_path,'best_model.pth')
    model = init_model(cfg, mode='eval', type=cfg.model_type, path=model_path, writer=SummaryWriter())
    model.to(device)
    # transform setting
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # eval datasets
    eval_dataset    = Deepfake_datasets(cfg.evaldata, transform=transform, use_DCT=cfg.use_DCT)
    eval_dataloader = build_dataloader(eval_dataset, batch_size=cfg.batch_size,num_workers=cfg.num_workers)
    # eval
    predict_all, label_all = [], []
    model.eval()
    with torch.no_grad():
        for data in tqdm(eval_dataloader):
            input, label = data[0].to(device), data[1].to(device)
            output = model(input)
            if type(output)==tuple:
                _,stu_cla = output[0],output[1]
            elif type(output)==torch.Tensor:
                stu_cla = output
            predict = stu_cla.data.cpu().numpy().argsort()[:,-1:].squeeze()
            predict_all.extend(predict.tolist())
            label_all.extend(label.cpu().numpy().tolist())
    # calculate acc, f1_score
    # print(type(label_all[0]),type(predict_all[0]))
    acc = accuracy_score(label_all,predict_all)
    f1  = f1_score(label_all,predict_all,average='macro')
    print(f"The accuracy is {acc}, f1_score is {f1}")
    # calculate confusion matrix
    matrix = confusion_matrix(label_all,predict_all,labels=[0,1,2])
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    sns.heatmap(matrix/np.sum(matrix, axis=1).reshape(3,1), annot=True, fmt='.2%', cmap="YlGnBu", linewidths=.5, xticklabels=[0,1,2], yticklabels=[0,1,2],ax=ax)
    ax.set_xlabel(cfg.model_save_path)
    cm_path = os.path.join(cfg.model_save_path, 'confusion_matrix.png')
    plt.savefig(cm_path)
    # write result in json
    if args.out:
        with open(os.path.join(cfg.model_save_path,args.out), 'w+') as f:
            json.dump({'label':label_all, 'pred':predict_all}, f)
    print("Finish evaluation.~")

if __name__=="__main__":
    eval()


'''
command take-away note:
python eval.py configs/resnet.py --out out.json
'''

