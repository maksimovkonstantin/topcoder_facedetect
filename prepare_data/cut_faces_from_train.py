import cv2
import os
import pandas as pd
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='parse input data')
    parser.add_argument('train_path', help='train path')
    args = parser.parse_args()
    train_path = args.train_path

    submit_file = os.path.join(train_path, 'training.csv')
    predictions = pd.read_csv(submit_file)
    test_images_path = args.train_path
    save_path = '/wdata/faces_cuts_train'
    os.mkdir(save_path)

    all_files = os.listdir(test_images_path)
    all_files = sorted([os.path.join(test_images_path, el) for el in all_files if el.split('.')[-1] == 'jpg' ])

    for _file in tqdm(all_files[:]):
        image = cv2.imread(_file)
        image_id = _file.split('/')[-1].split('.')[0]
        subdf = predictions[predictions['FILE'] == _file.split('/')[-1]]
        subdf = subdf.reset_index(drop=True)
        for i, row in tqdm(subdf.iterrows()):
            _, image_name, subject_id, x1, y1, w, h = row
            x1 = int(x1)
            y1 = int(y1)
            w = int(w)
            h = int(h)
            save_image_path = os.path.join(save_path, '{}_{}_{}.jpg'.format(subject_id, image_id, i))
            save_image = image[y1:y1 + h, x1:x1 + w, :]
            cv2.imwrite(save_image_path, save_image)


if __name__ == '__main__':
    main()

