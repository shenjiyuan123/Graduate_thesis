seed = 10
use_DCT = True

traindata = '/home/shenjiyuan/deepfake/deepfake/datasets/train_data_2.txt'
evaldata  = '/home/shenjiyuan/deepfake/deepfake/datasets/eval_data_2.txt'

epoch_num = 15
num_workers = 8
gpu_nums  = 4
batch_size = 128 * gpu_nums
log_interval = 20
evaluation_interval = 1

lr = 0.02
weight_decay=0.0001

pretrained_path = "/home/shenjiyuan/deepfake/deepfake/checkpoint/resnet18/resnet18_pretrained.pth"
model_save_path = "/home/shenjiyuan/deepfake/deepfake/checkpoint/resnet18/3/"
log_file = "/home/shenjiyuan/deepfake/deepfake/checkpoint/resnet18/3/logger.log"