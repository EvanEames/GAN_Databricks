# Databricks notebook source
# MAGIC %md <a href="https://colab.research.google.com/github/bkkaggle/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# COMMAND ----------

# MAGIC %md # Install

# COMMAND ----------

!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

# COMMAND ----------

import os
os.chdir('pytorch-CycleGAN-and-pix2pix/')

# COMMAND ----------

!pip install -r requirements.txt

# COMMAND ----------

# MAGIC %md # Datasets
# MAGIC 
# MAGIC Download one of the official datasets with:
# MAGIC 
# MAGIC -   `bash ./datasets/download_pix2pix_dataset.sh [cityscapes, night2day, edges2handbags, edges2shoes, facades, maps]`
# MAGIC 
# MAGIC Or use your own dataset by creating the appropriate folders and adding in the images. Follow the instructions [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md#pix2pix-datasets).

# COMMAND ----------

!bash ./datasets/download_pix2pix_dataset.sh facades

# COMMAND ----------

# MAGIC %md # Pretrained models
# MAGIC 
# MAGIC Download one of the official pretrained models with:
# MAGIC 
# MAGIC -   `bash ./scripts/download_pix2pix_model.sh [edges2shoes, sat2map, map2sat, facades_label2photo, and day2night]`
# MAGIC 
# MAGIC Or add your own pretrained model to `./checkpoints/{NAME}_pretrained/latest_net_G.pt`

# COMMAND ----------

!bash ./scripts/download_pix2pix_model.sh facades_label2photo

# COMMAND ----------

# MAGIC %md # Training
# MAGIC 
# MAGIC -   `python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA`
# MAGIC 
# MAGIC Change the `--dataroot` and `--name` to your own dataset's path and model's name. Use `--gpu_ids 0,1,..` to train on multiple GPUs and `--batch_size` to change the batch size. Add `--direction BtoA` if you want to train a model to transfrom from class B to A.

# COMMAND ----------

!python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

# COMMAND ----------

# MAGIC %md # Testing
# MAGIC 
# MAGIC -   `python test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name facades_pix2pix`
# MAGIC 
# MAGIC Change the `--dataroot`, `--name`, and `--direction` to be consistent with your trained model's configuration and how you want to transform images.
# MAGIC 
# MAGIC > from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix:
# MAGIC > Note that we specified --direction BtoA as Facades dataset's A to B direction is photos to labels.
# MAGIC 
# MAGIC > If you would like to apply a pre-trained model to a collection of input images (rather than image pairs), please use --model test option. See ./scripts/test_single.sh for how to apply a model to Facade label maps (stored in the directory facades/testB).
# MAGIC 
# MAGIC > See a list of currently available models at ./scripts/download_pix2pix_model.sh

# COMMAND ----------

!ls checkpoints/

# COMMAND ----------

!python test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name facades_label2photo_pretrained

# COMMAND ----------

# MAGIC %md # Visualize

# COMMAND ----------

import matplotlib.pyplot as plt

img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_fake_B.png')
plt.imshow(img)

# COMMAND ----------

img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_real_A.png')
plt.imshow(img)

# COMMAND ----------

img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_real_B.png')
plt.imshow(img)