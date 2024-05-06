import json
import numpy as np
import pandas as pd

def preprocess_ml25m(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[3]) - 789652009
      label = float(e[2])  # int(e[3])

      feat = np.array([float(e[2]),float(0)])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)

def reindex(df, data_name):
    new_df = df.copy()
    if data_name=='Flights' or data_name=='mooc' or data_name=='ml25m':
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df



def run(data_name):
    PATH = './data/{}_raw.csv'.format(data_name)
    OUT_DF = './data/{}.csv'.format(data_name)
    OUT_FEAT = './data/{}.npy'.format(data_name)
    df, feat = preprocess_ml25m(PATH)
    new_df = reindex(df, data_name)
    
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])
    
    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    
    

run('ml25m')