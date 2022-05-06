seed = 10
use_DCT = False
img_size = (224,224)

traindata = '/home/shenjiyuan/deepfake/deepfake/datasets/train_data_3.txt'
evaldata  = '/home/shenjiyuan/deepfake/deepfake/datasets/eval_data_3.txt'

epoch_num = 30
num_workers = 8
gpu_nums  = 4
batch_size = 128 * gpu_nums
log_interval = 20
evaluation_interval = 1

lr = 0.002
weight_decay=0.0001
betas=(0.9,0.999)

model_type = 'resnet'
pretrained_path = "/home/shenjiyuan/deepfake/deepfake/checkpoint/resnet18/resnet18_pretrained.pth"
model_save_path = "/home/shenjiyuan/deepfake/deepfake/checkpoint/resnet18/4/"
log_file = "/home/shenjiyuan/deepfake/deepfake/checkpoint/resnet18/4/logger.log"