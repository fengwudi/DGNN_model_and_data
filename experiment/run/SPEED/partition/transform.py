import pandas as pd
import argparse


def csv2txt(data):
    df = pd.read_csv(f'./data/ml_{data}.csv')
    df = df.drop(['Unnamed: 0', 'label', 'idx'], axis=1)
    df.to_csv(f'./partition/data/{data}.csv', index=False, header=False)

    fr = open(f'./partition/data/{data}.csv', 'rt')
    fw = open(f'./partition/data/{data}.txt', 'w+')

    ls = []

    for line in fr:
        line = line.replace('\n', '')
        line = line.split(',')
        ls.append(line)

    for row in ls:
        fw.write(' '.join(row) + '\n')

    fr.close()
    fw.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='wikipedia', help='dataset name')
    # parser.add_argument('--csv_file', type=str, default='./partition/data/ml_wikipedia.csv', help='csv file to trans')
    # parser.add_argument('--txt_file', type=str, default='./partition/data/wikipedia.txt', help='name and save the txt_file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    csv2txt(args.data)
