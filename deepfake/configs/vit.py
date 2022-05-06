seed = 10
use_DCT = True

traindata = '/home/shenjiyuan/deepfake/deepfake/datasets/train_data_3.txt'
evaldata  = '/home/shenjiyuan/deepfake/deepfake/datasets/eval_data_3.txt'

epoch_num = 30
num_workers = 8
gpu_nums  = 4
batch_size = 64 * gpu_nums
log_interval = 20
evaluation_interval = 1

lr = 0.02
weight_decay=0.01

pretrained_path = None
model_save_path = "/home/shenjiyuan/deepfake/deepfake/checkpoint/vit/2/"
log_file = "/home/shenjiyuan/deepfake/deepfake/checkpoint/vit/2/logger.log"