# Workflow
## Part 1
FFHQ datasets --> Real

GAN --> Fake & binary edited field

**Details:**

1. - [x] 27000 FFHQ -> Crop & Seg 

      **vs.** 

      69999 FFHQ -> raw img

2. - [x] each 10000 generated img -> by WGAN-GP, StyleGAN, MSPIE-StyleGANv2

3. - [x] Edited img 10000 -> by 1913 now

      Through Faceforensics, first detect the face, then splited the frames' and masks' box. (20frames extract 1 image)
      
      Fakeimgs: Face2Face(983), FaceSwap(784), DeepFakeDetection(146). All 1913 images.

4. - [x] DCT -> Spectrogram (have already seen)

    http://arxiv.org/abs/2003.08685
   
    https://arxiv.org/abs/2003.01826

*Notes:*

1. Background blank or not 

2. ~~StarGAN & SCFEGAN --> editable  --> not feasible~~

3. StyleGAN --> create from nothing

4. Seg or Crop use mediapipe selfie segmentation
  - [x] ~~check & read --> effect not good~~
  - [x] use it to crop the person —> ok

5. - [ ] Use DALL-E to generate editable img
   - [x] reconstruct the editable datasets from FaceForensics (each videos contain 20 or above frames)

### Datasets

**0:real, 1:fake, 2:editable**

#### 1

30500 train: 5000FFHQ; gen each 8000 equals to 24000; DFD 100, F2F 800, FS 600

8416 eval:  2000FFHQ; gen each 2000 equals to 6000; DFD 46, F2F 183, FS 184

#### 2

40500 train: 15000FFHQ; gen each 8000 equals to 24000; DFD 100, F2F 800, FS 600

11416 eval:  5000FFHQ; gen each 2000 equals to 6000; DFD 46, F2F 183, FS 184

#### 3 -> have leakage problem

FF++: delete DFD and sample 30frames for each videos ==> All have F2F 1122; FS 1162

40500 train: 15000FFHQ; gen each 8000 equals to 24000; F2F 800, FS 800

11416 eval:  5000FFHQ; gen each 2000 equals to 6000; F2F 323, FS 363

#### 4: train/eval_data_seg_2

- 由于os.listdir造成的数据泄漏问题，修正了数据集(seg_1包含泄漏问题)。
- 加入了mask
- 无mask的叫train/eval_data_4

FF++: delete DFD and sample 30frames for each videos ==> All have F2F 1122; FS 1162

40500 train: 15000FFHQ; gen each 8000 equals to 24000; F2F 794, FS 785

11416 eval:  5000FFHQ; gen each 2000 equals to 6000; F2F 328, FS 377

---
## Part 2
ResNet as the classification baseline --> 0 & 1 & 2

### Resnet:

#### 1 

https://github.com/shenjiyuan123/Graduate_thesis/tree/main/deepfake/checkpoint/resnet18/1

使用raw image进行训练，结果在logger里面，数据为train/eval_1

#### 2

https://github.com/shenjiyuan123/Graduate_thesis/tree/main/deepfake/checkpoint/resnet18/2

使用dct对raw image进行变换，数据为train/eval_1

#### 3

https://github.com/shenjiyuan123/Graduate_thesis/tree/main/deepfake/checkpoint/resnet18/3

使用dct对raw image进行变换，数据为train/eval_2

原因：由于1，2的结果对于real类别几乎全部判断为了1fake，并且在2中训练严重过拟合，因此考虑通过增加训练的real数据到15000。但效果也一般。

#### 4

https://github.com/shenjiyuan123/Graduate_thesis/tree/main/deepfake/checkpoint/resnet18/4

使用raw image，数据为train/eval_3 ==> f1: 72

后期发现是学习率设置偏大，以上实验设置都是0.02，后面改为了0.0002

#### 5

https://github.com/shenjiyuan123/Graduate_thesis/tree/main/deepfake/checkpoint/resnet18/5

使用raw image，数据为train/eval_3 ==> f1: 90

batch size为128

以此作为resnet18的baseline

#### 6

https://github.com/shenjiyuan123/Graduate_thesis/tree/main/deepfake/checkpoint/resnet18/6

使用raw image，数据为train/eval_3 ==> f1: 96

batch size为256

不准备以此作为resnet18的baseline

------

### F3Net:


|  Model   |  Paper   |        Valid(Mine)         |
| :------: | :------: | :------------------------: |
| Baseline |   89.3   |       F3Net test1:87       |
|   FAD    |   90.7   |            96.4            |
|   LFS    |   88.9   |           97.71            |
| ~~Both~~ | ~~92.8~~ | ~~F3Net test2:93(第二轮)~~ |
|   Mix    |   93.3   |       99.09(第二轮)        |
|  w/Seg   |          |           98.33            |

#### 1

数据为train/eval_3 

使用raw image, xception为baseline。第五轮得到87.

lr=0.002, bs=64

#### 2

数据为train/eval_3 

使用raw image，xception both。第二轮得到93分。

lr=0.002, bs=16

#### 3

数据为train/eval_3 

使用raw image，xception both,减少了4567四个block。第一轮70分，第二轮得到99分。

lr=0.002, bs=24

#### 4

数据为train/eval_3 

使用raw image，xception mix,减少了4567四个block。第一轮91分，第二轮得到99分。

lr=0.002, bs=24

~~*准备使用attention作为output mask*~~

#### 5

数据为train/eval_seg_2

multi-task，seg by using Decoder，

**Exp 4**的基础上，在第二次mixblocker后接反卷积得到224*224的mask.

采用loss = cls_loss + seg_bce + seg_dice

lr=0.002, bs=32

#### 6,7,8

bs=64

exp6:FAD

exp7:LFS

exp8:Origin





*Notes:*

1. Further reading: working~

- [x] https://zhuanlan.zhihu.com/p/376972313, https://zhuanlan.zhihu.com/p/378829258
- [x] **F3-Net:** https://arxiv.org/pdf/2007.09355.pdf

2. ~~Establish the basline model~~
3. …

---
## Part 3
Method:

1. Local Relation Learning for Face Forgery Detection
2. On the Detection of Digital Face Manipulation (use unsupervised learning)
3. RGB + facial landmark, Use two stream network to detect: https://arxiv.org/abs/1910.05455
4. …

## Part 4

1. why DCT, not DFT?
2. Visualize the middle feature map in order to see why CNN can make the perfect classification.
