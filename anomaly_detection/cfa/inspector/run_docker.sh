#!/bin/bash

sudo docker run \
--rm \
--name handok_cfa_inf \
-d \
-it \
--runtime=nvidia \
-e NVIDIA_VISIBLE_DEVICES=0 \
-v /home/spiderman/working/deep_learning/dockers/handok_cfa_inference:/torch \
-v /home/storage_disk2/datasets/handok/241223:/dataset \
-v /home/storage_disk2/datasets/handok/241223/ketotop/real_test:/test \
-v /usr:/usr \
handok_cfa_inf:latest
