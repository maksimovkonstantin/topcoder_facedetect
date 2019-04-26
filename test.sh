#!/usr/bin/env bash
python /project/prepare_data/check_pretrained.py
python /project/predict/create_submit.py $1 /wdata/solution_boxes.csv
python /project/prepare_data/cut_faces_from_test.py $1
python2.7 /project/insightface/deploy/get_test_features.py
python2.7 /project/predict/predict_knn.py $2