import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import pandas as pd 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from itertools import product
import random
import time


data_root = "D:\\sbv Microbiomics\\data\\" 

def random_cut(full_seq, sub_seq_len, overlap_coef):
    cut_indexes = random.sample(range(0, len(full_seq)-sub_seq_len), overlap_coef * int((len(full_seq))/sub_seq_len))
    seq_list = np.array([full_seq[i:i+sub_seq_len] for i in cut_indexes])
    return seq_list
    
    
def make_arrays(nb_rows, nb_features):
    if nb_rows:
        dataset = np.ndarray((nb_rows, nb_features), dtype=np.float32)
    else:
        dataset = None
    return dataset

def make_features(seq_list, sub_seq_bank):
    features_list  = make_arrays(len(seq_list), len(sub_seq_bank))
    for i, seq in enumerate(seq_list):    
        sub_seq_count = []
        for sub_seq in sub_seq_bank:
            sub_seq_count.append(seq.count(sub_seq))
        features_list[i] = sub_seq_count
    return features_list

def make_data_frame(features, lable):
    df = pd.DataFrame(features)
    df['lable'] = lable    
    return df

def make_sub_seq_bank(initial_string, sub_seq_len):
    return [''.join(tup) for tup in  list(set(product(set(initial_string), repeat = sub_seq_len)))]
   
    
def make_data_from_long_seq_list(long_seq_list):
    df_list = []
    sub_seq_bank = make_sub_seq_bank('ACTG', 4)
    for i, long_seq in enumerate(long_seq_list):
        seq_list = random_cut(long_seq, 151, 2)
        df_tmp = make_data_frame(make_features(seq_list, sub_seq_bank), i)
        df_list.append(df_tmp)
    return pd.concat(df_list)


if __name__ == '__main__':

    print("started")
    full_seq11 = []
    full_seq22 = []

    with open(data_root + "CBMB205.fasta", 'r') as myfile:
        full_seq11=myfile.read().replace('\n', '')
        
    with open(data_root + "F0441.fna", 'r') as myfile:
        full_seq22=myfile.read().replace('\n', '')
        
    full_seq1 = full_seq11[:1000000]
    full_seq2 = full_seq22[:1000000]
    full_seq3 = full_seq22[1000000:]
    
    long_seq_list = [full_seq1, full_seq2]
    df = make_data_from_long_seq_list(long_seq_list)

    df = df.sample(frac=1).reset_index(drop=True)
    X = df.values[:, :-1]
    y = df.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    start_time = time.time()
    clf = svm.SVC(C=5) #kernel='rbf', C=5, degree=2, tol=1e-7)
    clf.fit(x_train, y_train) 
    print("done")
    print("--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    y_pred = clf.predict(x_test)
    print("done")
    print("--- %s seconds ---" % (time.time() - start_time))

    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))




