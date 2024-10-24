from transformers import pipeline

from PIL import Image
import requests
import os
import numpy as np
import cv2
import torch

def visulize_depth(abs_depth, output_file):
    normalized_depth = (1 - ((abs_depth - abs_depth.min()) / (abs_depth.max() - abs_depth.min())))*255.0
    # normalized_depth = normalized_depth.numpy()
    cv2.imwrite(output_file, normalized_depth)


input_dir = "/home/raja/courses/csci677/HW3/gaussian-splatting/data/10/images"
output_dir = "/home/raja/courses/csci677/HW3/gaussian-splatting/data/10/abs_depth"
# input_dir = "./input"
# output_dir = "./output"
resolution = (1600, 1200)

checkpoint = "vinvino02/glpn-nyu"
depth_estimator = pipeline("depth-estimation", model=checkpoint)

for curr_image in os.listdir(input_dir):
    curr_image_path = os.path.join(input_dir, curr_image)
    file_name = curr_image.split('.')[0]
    print("processing", file_name)
    image = Image.open(curr_image_path)
    image = image.resize(resolution)
    print(image.size)
    predictions = depth_estimator(image)
    # absolute_depth = predictions["predicted_depth"][0]
    predicted_depth = predictions["predicted_depth"]
    
    
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    prediction = prediction.numpy()

    print(prediction.shape)

    output_npy_file = os.path.join(output_dir, file_name + '.npy')
    visu_file = os.path.join(output_dir, file_name + '.png')

    np.save(output_npy_file, prediction)
    visulize_depth(prediction, visu_file)