import face_model
import argparse
import cv2
import sys
import numpy as np
import os
import albumentations as A
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/wdata/pretrained/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=1, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--train_images_path', default='/wdata/faces_cuts_test', type=str, help='path to images`')

args = parser.parse_args()
model = face_model.FaceModel(args)

train_images_path = args.train_images_path
save_path = '/wdata/mxnet_test_features.pkl'
all_files = os.listdir(train_images_path)
all_files = [os.path.join(train_images_path, el) for el in all_files]

X = []
y = []
for _file in tqdm(all_files[:]):
  
    image = cv2.imread(_file)
    if image is None:
        continue
    img = model.get_input(image)
    aligned = img
    if aligned is None:
        continue

    embedding = model.get_feature(aligned)
    X.append(embedding)
    y.append(_file.split('/')[-1])
print(len(y))
res = {'X': X, 'y': y}
with open(save_path, 'wb') as f:
    pickle.dump(res, f)
