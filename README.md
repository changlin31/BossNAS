# BossNAS
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bossnas-exploring-hybrid-cnn-transformers/neural-architecture-search-on-imagenet)](https://paperswithcode.com/sota/neural-architecture-search-on-imagenet?p=bossnas-exploring-hybrid-cnn-transformers)                      
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bossnas-exploring-hybrid-cnn-transformers/neural-architecture-search-on-nats-bench-size-1)](https://paperswithcode.com/sota/neural-architecture-search-on-nats-bench-size-1?p=bossnas-exploring-hybrid-cnn-transformers)                     
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bossnas-exploring-hybrid-cnn-transformers/neural-architecture-search-on-nats-bench-size-2)](https://paperswithcode.com/sota/neural-architecture-search-on-nats-bench-size-2?p=bossnas-exploring-hybrid-cnn-transformers)

This repository contains PyTorch code and pretrained models of our paper: [***BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search***](https://arxiv.org/pdf/2103.12424.pdf) (ICCV 2021).

<p align="center">
<img src=https://user-images.githubusercontent.com/61453811/112087629-45c29700-8bc9-11eb-8536-3485660bc7c2.png width=95%/></p>
<p align="center">
Illustration of the Siamese supernets training with ensemble bootstrapping.
</p>

<p align="center"><img src=https://user-images.githubusercontent.com/61453811/112087643-4a874b00-8bc9-11eb-9440-757429034d81.png width=95%/></p>
<p align="center">
Illustration of the fabric-like Hybrid CNN-transformer Search Space with flexible down-sampling positions.
</p>

## Our Results and Trained Models

- Here is a summary of our searched models:

  |       Model       | MAdds | Steptime | Top-1 (%) | Top-5 (%) | Url           |
  | ----------------- | :---: | :------: | :-------: | :-------: | ------------- |
  | BossNet-T0 w/o SE | 3.4B  |  101ms   |   80.5    |   95.0    | [checkpoint](https://github.com/changlin31/BossNAS/releases/download/v0.1/BossNet-T0-nose-80_5.pth)   |
  |    BossNet-T0     | 3.4B  |  115ms   |   80.8    |   95.2    | [checkpoint](https://github.com/changlin31/BossNAS/releases/download/v0.1/BossNet-T0-80_8.pth)   |
  |    BossNet-T0^    | 5.7B  |  147ms   |   81.6    |   95.6    | same as above |
  |    BossNet-T1     | 7.9B  |  156ms   |   81.9    |   95.6    | [checkpoint](https://github.com/changlin31/BossNAS/releases/download/v0.1/BossNet-T1-81_9.pth)   |
  |    BossNet-T1^    | 10.5B |  165ms   |   82.2    |   95.7    | same as above |

- Here is a summary of architecture rating accuracy of our method:

  | Search space   | Dataset     | Kendall tau | Spearman rho | Pearson R |
  | -------------- | ----------- | :---------: | :----------: | :-------: |
  | MBConv         | ImageNet    | 0.65        | 0.78         | 0.85      |
  | NATS-Bench Ss  | Cifar10     | 0.53        | 0.73         | 0.72      |
  | NATS-Bench Ss  | Cifar100    | 0.59        | 0.76         | 0.79      |

## Usage

### 1. Requirements

- Linux
- Python 3.5+
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher

- Install [PyTorch](http://pytorch.org/) 1.7.0+ and torchvision 0.8.1+, for example:
  ```shell
  conda install -c pytorch pytorch torchvision
  ```
- Install [Apex](), for example:
  ```shell
  git clone https://github.com/NVIDIA/apex.git
  cd apex
  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```
- Install [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models), for example:
  ```shell
  pip install timm==0.3.2
  ```
- Install [OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup). As the original OpenSelfSup can not be installed as a site-package, please install our forked and modified version, for example:
  ```shell
  git clone https://github.com/changlin31/OpenSelfSup.git
  cd OpenSelfSup
  pip install -v --no-cache-dir .
  ```


- ImageNet & meta files
    - Download ImageNet from http://image-net.org/. Move validation images to labeled subfolders using following script: [https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.shvalprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
    - download imagenet meta files from [Google Drive](https://drive.google.com/drive/folders/1wYkJU_1qRHEt1LPVjBiG6ddUFV-t9hVJ?usp=sharing), put it under `/YOURDATAROOT/imagenet/`
  
- Download [NATS-Bench](https://github.com/D-X-Y/NATS-Bench) split version CIFAR datasets from [Google Drive](https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4). Put it under `/YOURDATAROOT/cifar/`

- Prepare BossNAS repository:
    ```shell
    git clone https://github.com/changlin31/BossNAS.git
    cd BossNAS
    ```

  - Create a soft link to your data root:
    ```shell
    ln -s /YOURDATAROOT data
    ```
  - Overall stucture of the folder:
    ```
    BossNAS
    ├── ranking_mbconv
    ├── ranking_nats
    ├── retraining_hytra
    ├── searching
    ├── data
    │   ├── imagenet
    │   │   ├── meta
    │   │   ├── train
    │   │   |   ├── n01440764
    │   │   |   ├── n01443537
    │   │   |   ├── ...
    │   │   ├── val
    │   │   |   ├── n01440764
    │   │   |   ├── n01443537
    │   │   |   ├── ...
    │   ├── cifar
    │   │   ├── cifar-10-batches-py
    │   │   ├── cifar-100-python
    ```

### 2. Retrain or Evaluate our BossNet-T models

- First, move to retraining code directory to perform Retraining or Evaluation.
    ```shell
    cd retraining_hytra
    ```
  Our retraining code of BossNet-T is based on [DeiT](https://github.com/facebookresearch/deit) repository.


- Evaluate our BossNet-T models with the following command:
  - Please download our checkpoint files from the result table, and change the `--resume` and `--input-size` accordingly. You can change the `--nproc_per_node` option to suit your GPU numbers

    ```shell
    python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model bossnet_T0 --input-size 224 --batch-size 128 --data-path ../data/imagenet --num_workers 8 --eval --resume PATH/TO/BossNet-T0-80_8.pth
    ```

    ```shell
    python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model bossnet_T1 --input-size 224 --batch-size 128 --data-path ../data/imagenet --num_workers 8 --eval --resume PATH/TO/BossNet-T1-81_9.pth
    ```

- Retrain our BossNet-T models with the following command:

  - You can change the `--nproc_per_node` to suit your GPU numbers. Please note that the learning rate will be automatically scaled according to the GPU numbers and batchsize. We recommend training with 128 batchsize and 8 GPUs. (takes about 2 days)

    ```shell
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model bossnet_T0 --input-size 224 --batch-size 128 --data-path ../data/imagenet --num_workers 8
    ```

    ```shell
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model bossnet_T1 --input-size 224 --batch-size 128 --data-path ../data/imagenet --num_workers 8
    ```

<p align="center"><img src=https://user-images.githubusercontent.com/61453811/112087617-40fde300-8bc9-11eb-93ed-d043979d3e65.png width=60%/></p>
<p align="center">Architecture of our BossNet-T0</p>

### 3. Evaluate architecture rating accuracy of BossNAS

- Get the ranking correlations of BossNAS on MBConv search space with the following commands:

    ```shell
    cd ranking_mbconv
    python get_model_score_mbconv.py
    ```

<p align="center"><img src=https://user-images.githubusercontent.com/61453811/112087625-43603d00-8bc9-11eb-8199-402998b9c7ef.png width=90%/></p>

- Get the ranking correlations of BossNAS on NATS-Bench Ss with the following commands:
    ```shell
    cd ranking_nats
    python get_model_score_nats.py
    ```

<p align="center"><img src=https://user-images.githubusercontent.com/61453811/112087637-48bd8780-8bc9-11eb-8697-ff535cc9634b.png width=20%/>
</p>

### 4. Search Architecture with BossNAS
First, go to the searching code directory:
```shell
cd searching
```

- Search in NATS-Bench Ss Search Space on CIFAR datasets (4 GPUs, 3 hrs)
  
  - CIFAR10:
    ```shell
    bash dist_train.sh configs/nats_c10_bs256_accumulate4_gpus4.py 4
    ```
  - CIFAR100:
    ```shell
    bash dist_train.sh configs/nats_c100_bs256_accumulate4_gpus4.py 4
    ```
  
- Search in MBConv Search Space on ImageNet (8 GPUs, 1.5 days)
  ```shell
  bash dist_train.sh configs/mbconv_bs64_accumulate8_ep6_multi_aug_gpus8.py 8
  ```
- Search in HyTra Search Space on ImageNet (8 GPUs, 4 days, memory requirement: 24G)
  ```shell
  bash dist_train.sh configs/hytra_bs64_accumulate8_ep6_multi_aug_gpus8.py 8
  ```

## Citation
If you use our code for your paper, please cite:
```bibtex
@inproceedings{li2021bossnas,
  author = {Li, Changlin and
            Tang, Tao and
            Wang, Guangrun and
            Peng, Jiefeng and
            Wang, Bing and
            Liang, Xiaodan and
            Chang, Xiaojun},
  title = {BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search},
  booktitle = {ICCV},
  year = 2021,
}
```
