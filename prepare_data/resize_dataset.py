import os
import cv2
import numpy as np
import pandas as pd
import mmcv
from tqdm import tqdm
import argparse


def main():
    cut_size = (1296*2, 864*2)
    parser = argparse.ArgumentParser(description='parse input data')
    parser.add_argument('train_path', help='train path')
    args = parser.parse_args()

    images_path = args.train_path
    images_save_path = '/wdata/remaked_dataset'
    os.mkdir(images_save_path)

    annot_path = os.path.join(images_path, 'training.csv')
    folds_path = '/wdata/folds.csv'
    
    folds = pd.read_csv(folds_path)
    all_train_files = sorted(folds[folds['fold'] != 0]['FILE'].tolist())
    all_val_files = sorted(folds[folds['fold'] == 0]['FILE'].tolist())
    annot_df = pd.read_csv(annot_path)
    annot_df['id'] = annot_df['FILE'].apply(lambda x: x.split('.')[0])

    all_train_ids = [el.split('.')[0] for el in all_train_files if el.split('.')[-1] == 'jpg']
    all_val_ids = [el.split('.')[0] for el in all_val_files if el.split('.')[-1] == 'jpg']

    for dataset_i, all_ids in enumerate([all_train_ids, all_val_ids]):
        annotations_result = []
        for _file in tqdm(all_ids[:]):
            file_annot = annot_df[annot_df['id'] == _file]

            if len(file_annot) == 0:
                continue

            image_path = os.path.join(images_path, '{}.jpg'.format(_file))
            image = mmcv.imread(image_path)
            h, w, c = image.shape
            image = cv2.resize(image, cut_size, interpolation=cv2.INTER_NEAREST)
            new_h, new_w, _ = image.shape
            cv2.imwrite(os.path.join(images_save_path, '{}.jpg'.format(_file)), image)
            w_scale = w / new_w
            h_scale = h / new_h

            annot_instance = {}
            annot_instance['filename'] = '{}.jpg'.format(_file)
            annot_instance['height'] = new_h
            annot_instance['width'] = new_w
            annot_instance['ann'] = {}

            boxes = []
            labels = []

            for label_index, label_row in file_annot.iterrows():
                class_label = 1
                x1 = max(int(label_row['FACE_X']), 0)
                y1 = max(int(label_row['FACE_Y']), 0)
                face_w = int(label_row['FACE_WIDTH'])
                face_h = int(label_row['FACE_HEIGHT'])

                x2 = x1 + face_w
                y2 = y1 + face_h

                x1 = int(x1 / w_scale)
                x2 = int(x2 / w_scale)
                y1 = int(y1 / h_scale)
                y2 = int(y2 / h_scale)

                box = [x1, y1, x2, y2]
                boxes.append(box)
                labels.append(class_label)
            annot_instance['ann']['bboxes'] = (np.array(boxes)).astype(np.float32)
            annot_instance['ann']['labels'] = (np.array(labels)).astype(np.int64)
            annot_instance['ann']['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
            annot_instance['ann']['labels_ignore'] = (np.array([])).astype(np.int64)
            annotations_result.append(annot_instance)
        if dataset_i == 0:
            save_path = '/wdata/train_mmstyle_annotations.pkl'
        else:
            save_path = '/wdata/val_mmstyle_annotations.pkl'
        mmcv.dump(annotations_result, save_path)


if __name__ == '__main__':
    main()
    