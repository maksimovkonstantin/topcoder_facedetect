#!/usr/bin/env bash
python /project/prepare_data/clear_wdata.py
python /project/prepare_data/create_folds.py $1
python /project/prepare_data/resize_dataset.py $1
python /project/prepare_data/cut_faces_from_train.py $1
python2.7 /project/insightface/deploy/get_train_features.py
python2.7 /project/prepare_data/train_knn_on_all.py
/mmdetection/tools/dist_train.sh /project/configs/final_config.py 1 \
--work_dir /wdata/train_logs \
--seed 769 \
--validate