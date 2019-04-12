# CANZSL ACM MM
code for the paper.


Data:
You can download the dataset [CUBird and NABird](https://drive.google.com/open?id=1YUcYHgv4HceHOzza8OGzMp092taKAAq1)   
Put the uncompressed data to the folder "data"

## Reproduce results 
#### CUBird SCS mode && SCE mode
```shell
python train.py --dataset CUB2011 --splitmode easy
python train.py --dataset CUB2011 --splitmode hard
```

#### NABird SCS mode && SCE mode
```shell
python train.py --dataset NABird --splitmode easy
python train.py --dataset NABird --splitmode hard
```

Reference:

[1] Yizhe Zhu, Mohamed Elhoseiny, Bingchen Liu, Xi Peng and Ahmed Elgammal
"A Generative Adversarial  Approach for Zero-Shot Learning from Noisy Texts", CVPR, 2018
