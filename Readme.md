# Anomaly-Dacon
[MVtec AD dataset Anomaly Detection in DACON.](https://dacon.io/competitions/official/235894/overview/description)<br>
Private 21th, Score 0.86967 (21/481, 4.4%) <br>
Write a save model name in args.comment

## Development Environment
* Ubuntu 20.04.3 LTS
* Intel i7-9700KF
* RTX2080Ti x 2
* CUDA 11.4
* Pytorch 1.11.0

## Train
* Model: ConvNext Base
* Image Size (384x384)
* Focal Loss
* AdamW Optimizer(2e-4) with warmup 2 epoch
* Train for 25 epochs
* 5 Stratified K Fold train
* Cutmix stop after 20 epoch
* Good Label (data size: large): weak augmentation
* Anomaly Label (data size: small): strong augmentation

## Inference
5 fold ensemble (soft voting)