import os
import pandas as pd
import sklearn.model_selection as ms
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description='parse input data')
    parser.add_argument('train_path', help='train path')
    args = parser.parse_args()

    seed = 769
    n_splits = 5
    train_path = args.train_path
    if os.path.exists(train_path):
        files = os.listdir(train_path)
        files = [el for el in files if el.split('.')[-1] == 'jpg']
        X = np.array(files)
        kf = ms.KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds_file = {'FILE': [], 'fold': []}
        for split_n, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            files = list(X_test)
            print(len(files))
            folds_file['FILE'] += files[:]
            folds_file['fold'] += [split_n]*len(files)
        folds_file = pd.DataFrame(folds_file)
        save_path = '/wdata/folds.csv'
        folds_file.to_csv(save_path, index=False)
    else:
        print('Invalid path')


if __name__ == '__main__':
    main()
    