# MIRM
### Codes for the paper "Improved multiple-image-based reflection removal algorithm using deep neural networks"
[[paper](https://ieeexplore.ieee.org/abstract/document/9234008)] 
![cover](cover.png)

### Codes are implemented on Pytorch>=1.0 

## How to use

### Training:

Prepare the images with slight shifts (light filed images) into the './scenes_train' folder for reflection image synthesizing and training the networks. We only use **five** of each group of images to generate small overlapped npy patch for speeding up the training process. This is implemented by 

```
python npy_save_database_5views.py
```

All the npy files will be stored in the 'info_four_closest_corners_train_set'  folder. Then

- Train the disparity network: 

```
python train_disparity.py
```

- Train the edge reconstruction network: 
```
python train_edge.py --train_label_dir info_four_closest_corners_train_set
```
- Train the image reconstruction network: 
```
python train_img_rec.py --train_label_dir info_four_closest_corners_train_set
```
### Inference and evaluation:

- We also provide the <u>[pre-trained models](https://drive.google.com/file/d/1UmwgggXnpxeql4ZFi3Vq9Y_vgvMXyFxV/view?usp=sharing)</u> and the <u>[synthesized test data](https://drive.google.com/file/d/15JF9PMc0oCxwA-ZoCuE-werjDcj0LS4k/view?usp=sharing)</u> for evaluation. 
```
python image_separation.py --test_imgs_folder ... --model_dir ...
```
## Citation
T. Li, Y.-H. Chan, and D.P.K. Lun. "Improved multiple-image-based reflection removal algorithm using deep neural networks." IEEE Transactions on Image Processing, 2020.