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
   - [ ] reconstruct the editable datasets from FaceForensics (each videos contain 20 or above frames)

### Datasets

**0:real, 1:fake, 2:editable**

#### 1

30500 train: 5000FFHQ; gen each 8000 equals to 24000; DFD 100, F2F 800, FS 600

8416 eval:  2000FFHQ; gen each 2000 equals to 6000; DFD 46, F2F 183, FS 184

#### 2

40500 train: 15000FFHQ; gen each 8000 equals to 24000; DFD 100, F2F 800, FS 600

11416 eval:  5000FFHQ; gen each 2000 equals to 6000; DFD 46, F2F 183, FS 184

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



*Notes:*

1. Further reading:

- [ ] https://zhuanlan.zhihu.com/p/376972313, https://zhuanlan.zhihu.com/p/378829258
- [ ] **F3-Net:** https://arxiv.org/pdf/2007.09355.pdf

2. Establish the basline model
3. …

---
## Part 3
Use two stream network to detect: https://arxiv.org/abs/1910.05455

RGB + facial landmark



