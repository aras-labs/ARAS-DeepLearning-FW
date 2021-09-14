#!/bin/bash
cd ..
echo "FasterRCNN-mobilenet_v3_large_320_fpn"
python3 train_SR.py "./config/capsulorhexis/FasterRCNN-mobilenet_v3_large_320_fpn.yml"

echo "FasterRCNN-mobilenet_v3_large_fpn"
python3 train_SR.py "./config/capsulorhexis/FasterRCNN-mobilenet_v3_large_fpn.yml"

echo "FasterRCNN-resnet50_fpn"
python3 train_SR.py "./config/capsulorhexis/FasterRCNN-resnet50_fpn.yml"

echo "SSDlite320_mobilenet_v3_large"
python3 train_SR.py "./config/capsulorhexis/SSDlite320_mobilenet_v3_large.yml"

echo "SSD300_vgg16"
python3 train_SR.py "./config/capsulorhexis/SSD300_vgg16.yml"

echo "RetinaNet_resnet50_fpn"
python3 train_SR.py "./config/capsulorhexis/RetinaNet_resnet50_fpn.yml"

