import pickle
import pandas as pd
import argparse
from tqdm import tqdm

def ss(f):
    return "{0}".format(f)

def main():
    parser = argparse.ArgumentParser(description='parse input data')
    parser.add_argument('result_file', help='result_file')
    args = parser.parse_args()

    model_path = '/wdata/pretrained/model_original_1.pkl'
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    data_path = '/wdata/mxnet_test_features.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)


    submit_file = '/wdata/solution_boxes.csv'
    predictions = pd.read_csv(submit_file, header=None)
    X = data['X']
    y = data['y']

    outFileName = args.result_file

    outFile = open(outFileName, "w")
    for i, el in enumerate(tqdm(X[:])):
        to_predict = el.reshape(1, -1)
        distance = knn_clf.kneighbors(to_predict, n_neighbors=1)[0][0][0]
        if distance > 0.9:
            continue
        res = knn_clf.predict(to_predict)[0]
        if res == -1:
            continue
        # print(y[i])
        _id = y[i].split('.')[0].split('_')[0] + '.jpg'
        number = int(y[i].split('.')[0].split('_')[-1])
        subdf = predictions[predictions[0] == _id]
        subdf = subdf.loc[subdf.index == number]; subdf = subdf.iloc[0, :]
        img_name, x1, y1, w, h, prob = subdf

        line = img_name + "," + ss(res) + "," + ss(x1) + "," + ss(y1) + "," + ss(w) + "," + ss(h) + "," + ss(prob) + "\n"
        outFile.write(line)
    outFile.close()


if __name__ == '__main__':
    main()