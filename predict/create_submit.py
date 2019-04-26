import os
import cv2
import mmcv
import numpy as np
import argparse
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from tqdm import tqdm
from mmdet.ops.nms import soft_nms


def ss(f):
    return "{0}".format(f)


def main():
    parser = argparse.ArgumentParser(description='parse input data')
    parser.add_argument('test_path', help='test path')
    parser.add_argument('result_file', help='result_file')
    args = parser.parse_args()

    test_path = args.test_path
    result_file = args.result_file
    config_path = '/project/configs/final_config.py'
    weights_path = '/wdata/train_logs/epoch_10.pth'
    nms_trs = 0.8
    
    all_files = sorted(os.listdir(test_path))
    all_files = [el for el in all_files if el.split('.')[-1]=='jpg']
    submit_dict = {}

    outFile = open(result_file, "w")
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None

    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, weights_path)

    for _file in tqdm(all_files[:]):
        file_path = os.path.join(test_path, _file)
        bboxes = []
        img = mmcv.imread(file_path)
        h, w, c = img.shape

        result = inference_detector(model, img, cfg)
        for det_i in range(result[0].shape[0]):
            x1, y1, x2, y2, prob = list(result[0][det_i])
            bboxes.append([x1, y1, x2, y2, prob])
        flipped = np.fliplr(img)
        result = inference_detector(model, flipped, cfg)
        for det_i in range(result[0].shape[0]):
            x1, y1, x2, y2, prob = list(result[0][det_i])
            bboxes.append([w - x2, y1, w - x1, y2, prob])

        bboxes = np.array(bboxes, dtype=np.float32)

        if len(bboxes) > 0:
            bboxes = np.array(bboxes, dtype=np.float32)
            bboxes = soft_nms(bboxes, nms_trs)
            bboxes = bboxes[0]
        else:
            continue
        bboxes = np.array(bboxes, dtype=np.float32)
        bboxes = np.vstack(bboxes)
        for det_i, det in enumerate(bboxes):
            if det_i >= 59:
                continue

            x1, y1, x2, y2, prob = det
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            w = x2 - x1
            h = y2 - y1
            if prob < 0.0:
                continue
            line = _file + "," + ss(x1) + "," + ss(y1) + "," + ss(w) + "," + ss(h) + "," + ss(prob) + "\n"
            outFile.write(line)
    outFile.close()


if __name__ == '__main__':
    main()