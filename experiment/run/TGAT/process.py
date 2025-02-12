import json
import numpy as np
import pandas as pd

def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    
    # print(data_name)
    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            ts = float(e[2])
            label = int(e[3])
            if data_name=='./processed/Flights.csv' or data_name=='./processed/mooc.csv': 
                feat = np.zeros(172)
            else:
                feat = np.array([float(x) for x in e[4:]])
            
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            
            feat_l.append(feat)
    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list}), np.array(feat_l)

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

      feat = np.array(float(e[2]))

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
    if data_name=='Flights' or data_name=='ml25m':
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
    PATH = './processed/{}.csv'.format(data_name)
    OUT_DF = './processed/ml_{}.csv'.format(data_name)
    OUT_FEAT = './processed/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './processed/ml_{}_node.npy'.format(data_name)
    if data_name=='ml25m':
        df, feat = preprocess_ml25m(PATH)
        feat = feat.reshape(-1,1)
    else:
        df, feat = preprocess(PATH)
    new_df = reindex(df, data_name)
    
    print(feat.shape)
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])
    
    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
    
    print(feat.shape)
    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)
    
    
# run('wikipedia')
# run('reddit')
# run('mooc')
# run('Flights')
# run('mooc')
run('ml25m')