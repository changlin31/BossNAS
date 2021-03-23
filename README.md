# BossNAS

This repository contains PyTorch evaluation code, retraining code and pretrained models of our paper: ***BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search***.

<img src=https://user-images.githubusercontent.com/61453811/112087629-45c29700-8bc9-11eb-8536-3485660bc7c2.png width=90%/>

Illustration of the Siamese supernets training with ensemble bootstrapping.



<img src=https://user-images.githubusercontent.com/61453811/112087643-4a874b00-8bc9-11eb-9440-757429034d81.png width=90%/>

Illustration of the fabric-like Hybrid CNN-transformer Search Space with flexible down-sampling positions.

## Our Results and Trained Models

- Here is a summary of our searched models:

  |       Model       | MAdds | Steptime | Top-1 (%) | Top-5 (%) | Url           |
  | :---------------: | :---: | :------: | :-------: | :-------: | :-----:       |
  | BossNet-T0 w/o SE | 3.4B  |  101ms   |   80.5    |   95.0    | coming soon   |
  |    BossNet-T0     | 3.4B  |  115ms   |   80.8    |   95.2    | coming soon   |
  |    BossNet-T0^    | 5.7B  |  147ms   |   81.6    |   95.6    | same as above |
  |    BossNet-T1     | 7.9B  |  156ms   |   81.9    |   95.6    | coming soon   |
  |    BossNet-T1^    | 10.5B |  165ms   |   82.2    |   95.7    | same as above |

- Here is a summary of architecture rating accuracy of our method:

  | Search space   | Dataset     | Kendall tau | Spearman rho | Pearson R |
  | :------------: | :---------: | ----------- | ------------ | --------- |
  | MBConv         | ImageNet    | 0.65        | 0.78         | 0.85      |
  | NATS-Bench Ss  | Cifar10     | 0.53        | 0.73         | 0.72      |
  | NATS-Bench Ss  | Cifar100    | 0.59        | 0.76         | 0.79      |

## Usage

### 1. Requirements

- Install [PyTorch](http://pytorch.org/) 1.7.0+ and torchvision 0.8.1+, for example:

```
conda install -c pytorch pytorch torchvision
```

- Install [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models), for example:

```
pip install timm==0.3.2
```

- Download the ImageNet dataset from http://image-net.org/, and move validation images to labeled subfolders

    - To do this, you can use the following script: [https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.shvalprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

### 2. Retrain or Evaluate our BossNet-T models

- First, move to retraining code directory to perform Retraining or Evaluation.
    ```bash
    cd HyTra_retraining
    ```
  Our retraining code of BossNet-T is based on [DeiT](https://github.com/facebookresearch/deit) repository.


- You can evaluate our BossNet-T models with the following command:

    ```bash
    python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model bossnet_T0 --input-size 224 --batch-size 128 --data-path /PATH/TO/ImageNet --num_workers 8 --eval --resume PATH/TO/BossNet-T0-80_8.pth
    ```

    ```bash
    python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model bossnet_T1 --input-size 224 --batch-size 128 --data-path /PATH/TO/ImageNet --num_workers 8 --eval --resume PATH/TO/BossNet-T1-81_9.pth
    ```
  Please download our checkpoint files from the result table. Please change the `--nproc_per_node` option to suit your GPU numbers, and change the `--data-path`, `--resume` and `--input-size` accordingly.


- You can retrain our BossNet-T models with the following command:

  Please change the `--nproc_per_node` and `--data-path` accordingly. Note that the learning rate will be automatically scaled according to the GPU numbers and batchsize. We recommend training with 128 batchsize and 8 GPUs.

    ```bash
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model bossnet_T0 --input-size 224 --batch-size 128 --data-path /PATH/TO/ImageNet --num_workers 8
    ```

    ```bash
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model bossnet_T1 --input-size 224 --batch-size 128 --data-path /PATH/TO/ImageNet --num_workers 8
    ```

<img src=https://user-images.githubusercontent.com/61453811/112087617-40fde300-8bc9-11eb-93ed-d043979d3e65.png width=60%/>

Architecture of our BossNet-T0

### 3. Evaluate architecture rating accuracy of BossNAS

- You can get the ranking correlations of BossNAS on MBConv search space with the following commands:

    ```bash
    cd MBConv_ranking
    python get_model_score_mbconv.py
    ```

<img src=https://user-images.githubusercontent.com/61453811/112087625-43603d00-8bc9-11eb-8199-402998b9c7ef.png width=90%/>

- You can get the ranking correlations of BossNAS on NATS-Bench Ss with the following commands:
    ```bash
    cd NATS_SS_ranking
    python get_model_score_nats.py
    ```

<img src=https://user-images.githubusercontent.com/61453811/112087637-48bd8780-8bc9-11eb-8697-ff535cc9634b.png width=20%/>

## TODO

Searching code will be released later.

