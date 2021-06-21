# maxgrad

This repository contains the source code accompanying our CVPR 2021 paper.

**[Gradient-Based Algorithms for Machine Teaching](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Gradient-Based_Algorithms_for_Machine_Teaching_CVPR_2021_paper.html)**  
[Pei Wang](http://www.svcl.ucsd.edu/~peiwang), [Kabir Nagrecha](https://knagrecha.github.io/), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno).  
In CVPR, 2021.

```
@InProceedings{wang2021gradient,
author = {Wang, Pei and Nagrecha, Kabir and Vasconcelos, Nuno},
title = {Gradient-Based Algorithms for Machine Teaching},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}
}
```

## Requirements

1. The project was implemented and tested in Python 3.5 and Pytorch 1.0. Other versions should work after minor modification.
2. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.


## Datasets

[Butterflies and Chinese Characters](https://github.com/macaodha/explain_teach/tree/master/data) are used. Please organize them as below after download,


```
datasets
|_ butterflies_crop
  |_ images
    |_ Viceroy
    |_ ...
```

```
datasets
|_ chinese_chars
  |_ images
    |_ grass
    |_ ...
```

## Implementation details

### To reproduce results of our method on simulated learners
```
train_butterflies_maxgrad.py
train_chineseChars_maxgrad.py
```

### Our selected teaching images are contained in
```
butterflies_Lt_gt_tr.txt
ChineseChars_Lt_gt_tr.txt
```

## Time and Space

All experiments were run on NVIDIA TITAN Xp 

1. butterflies

model     | #GPUs | train time |
---------|--------|-----|
train_butterflies_maxgrad     | 1 | ~5min    | 

2. Chinese characters


model     | #GPUs | train time |
---------|--------|-----|
train_chineseChars_maxgrad     | 1 | ~3min    | 


## Teaching system

[system](https://machine-teaching-website.herokuapp.com/tokensignin?username=1234)

## Disclaimer

For questions, feel free to reach out
```
Pei Wang: peiwang062@gmail.com
```


