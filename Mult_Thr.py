import sys
from multiprocessing import Pool, cpu_count
import numpy as np
import itertools
import pandas as pd 
from itertools import product
import random
from six.moves import cPickle as pickle
import gc
import time

def make_sub_seq_bank(initial_string, sub_seq_len):
    return [''.join(tup) for tup in  list(set(product(set(initial_string), repeat = sub_seq_len)))]

 
if __name__ == '__main__':

    print("started" )

    s1 = 'GACGAGTCGGCGGCGGCCGCGGCCGCCGCCTGCTTCGAGCAGGCCGCACAAATGTGGGCCGACGAGCGCGATGCGATCGATGCGCTGCTGCGCGCCGCGCAGCCGGCGCTCAACCAGCGCTCGCACAAGCCCGAGGCGATCGCCGATGCGT'
    s2 = 'GCGTGAACCCGAGCTATTCGCCGCCGCAGGTGATCCGCGGGCTTGCCGCCCGCTTGCCCGACGAGCGCCGCTGGGCCGCGCTGATGACGAGCACCGGCCGCGTGCTGCTCGACACCGCACCGAAGGGCTTCGCGCCGGACTGGGCGCTGTA' 
    seq_list = []
    sub_seq_bank = make_sub_seq_bank('ACTG', 4)

    for i in range(1000):
        seq_list.append(s1)
        seq_list.append(s2)
 
    # Build a pool of 8 processes
    pool = Pool(processes=cpu_count())

    # Collapse the lists of tuples into total term frequencies
    # term_frequencies = pool.map(Reduce, token_to_tuples.items())

    data_list = []
    start_time = time.time()
    for i, seq in enumerate(seq_list):
        sub_seq_count = list(pool.map(seq.count, sub_seq_bank))
        sub_seq_count.append("A"*(i+1))
        data_list.append(sub_seq_count)
    
    print("done parallel")
    print("--- %s seconds ---" % (time.time() - start_time))   
    print(len(data_list))

    data_list = []
    start_time = time.time()
    for i, seq in enumerate(seq_list):
        sub_seq_count = list(map(seq.count, sub_seq_bank))
        sub_seq_count.append("A"*(i+1))
        data_list.append(sub_seq_count)
    
    print("done single")
    print("--- %s seconds ---" % (time.time() - start_time))   
    print(len(data_list))

 
