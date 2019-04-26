import cv2
import os
import argparse
import pandas as pd
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='parse input data')
    parser.add_argument('train_path', help='train path')
    args = parser.parse_args()
    train_path = args.train_path

    submit_file = '/wdata/solution_boxes.csv'
    predictions = pd.read_csv(submit_file, header=None)
    test_images_path = train_path
    save_path = '/wdata/faces_cuts_test'
    os.mkdir(save_path)
    
    all_files = os.listdir(test_images_path)
    all_files = sorted([os.path.join(test_images_path, el) for el in all_files])
    for _file in tqdm(all_files[:]):
        image = cv2.imread(_file)
        subdf = predictions[predictions[0] == _file.split('/')[-1]]
        for i, row in tqdm(subdf.iterrows()):
            image_name, x1, y1, w, h, _ = row
            save_image_path = os.path.join(save_path, '{}_{}.jpg'.format(_file.split('/')[-1].split('.')[0], i))
            save_image = image[y1:y1 + h, x1:x1 + w, :]
            cv2.imwrite(save_image_path, save_image)


if __name__ == '__main__':
    main()
