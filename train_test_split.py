import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

# split dataset, input a .csv file, split it into two separate datasets.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, default='./data_set.csv', help='path to the .csv file',
                        required=False)

    parser.add_argument('--train_dir', type=str, default='./data_train.csv', help='path to the train .csv file',
                        required=False)
    parser.add_argument('--test_dir', type=str, default='./data_test.csv', help='path to the test .csv file',
                        required=False)

    parser.add_argument('--test_size', type=float, default=0.1, help='test size',
                        required=False)
    parser.add_argument('--seed', type=int, default=42, help='random seed',
                        required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    df = pd.read_csv(args.df_path)

    # remove empty samples
    df = df[df['tree_id'] != '']

    # input and target
    X = df.iloc[:, :625]
    y = df.iloc[:, 625:]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=args.seed)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_val, y_val], axis=1)

    # save .csv file
    df_train.to_csv(args.train_dir, index=False)
    df_test.to_csv(args.test_dir, index=False)