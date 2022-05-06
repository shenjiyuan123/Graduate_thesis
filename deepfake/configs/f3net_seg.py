seed = 10
use_DCT = False
img_size = (224,224)

traindata = '/home/shenjiyuan/deepfake/deepfake/datasets/a.txt'
# evaldata  = '/home/shenjiyuan/deepfake/deepfake/datasets/eval_data_seg_2.txt'
evaldata  = '/home/shenjiyuan/deepfake/deepfake/datasets/b.txt'

epoch_num = 30
num_workers = 8
gpu_nums  = 3
batch_size = 20 * gpu_nums
log_interval = 20
evaluation_interval = 1

lr = 0.002
weight_decay=0.0001
betas=(0.9,0.999)

multi_gpu = True
model_type = 'F3Net_Seg'
pretrained_path = None
model_save_path = "/home/shenjiyuan/deepfake/deepfake/checkpoint/F3Net/6/"
log_file = "/home/shenjiyuan/deepfake/deepfake/checkpoint/F3Net/6/logger.log"