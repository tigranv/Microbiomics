import sys
import numpy as np
import itertools
import pandas as pd 
from itertools import product
import random
from six.moves import cPickle as pickle
import gc
from multiprocessing import Process, Queue
import difflib, random, time
from concurrent.futures import ProcessPoolExecutor

def make_sub_seq_bank(initial_string, sub_seq_len):
    return [''.join(tup) for tup in  list(set(product(set(initial_string), repeat = sub_seq_len)))]

# def do_job(seq):
#     sub_seq_count = list(map(seq.count, sub_seq_bank))
#     sub_seq_count.append("A")
#     return sub_seq_count

# def multiprocessing(func, seq_list, workers):
#     with ProcessPoolExecutor(max_workers=workers) as executor:
#         res = executor.map(func, seq_list)
#     return list(res)




def do_job2(seq_list, bank:list, data_list):
    for i, seq in enumerate(seq_list):
        sub_seq_count = list(map(seq.count, bank))
        sub_seq_count.append("A"*(i+1))
        data_list.append(sub_seq_count)
    return data_list




if __name__ == '__main__':

    print("started" )

    s1 = 'GACGAGTCGGCGGCGGCCGCGGCCGCCGCCTGCTTCGAGCAGGCCGCACAAATGTGGGCCGACGAGCGCGATGCGATCGATGCGCTGCTGCGCGCCGCGCAGCCGGCGCTCAACCAGCGCTCGCACAAGCCCGAGGCGATCGCCGATGCGT'
    s2 = 'GCGTGAACCCGAGCTATTCGCCGCCGCAGGTGATCCGCGGGCTTGCCGCCCGCTTGCCCGACGAGCGCCGCTGGGCCGCGCTGATGACGAGCACCGGCCGCGTGCTGCTCGACACCGCACCGAAGGGCTTCGCGCCGGACTGGGCGCTGTA' 
    seq_list = []
    

    for i in range(10000):
        seq_list.append(s1)
        seq_list.append(s2)

    sub_seq_bank = make_sub_seq_bank('ACTG', 4)

    data_list = []
    start_time = time.time()
    #----------------------------------------------------------------------------------

    
    #split work into 5 or 10 processes
    processes = 4
    def splitlist(inlist, chunksize):
        return [inlist[x:x+chunksize] for x in range(0, len(inlist), chunksize)]

    print(len(seq_list)/processes)
    mainwordlistsplitted = splitlist(seq_list, int(len(seq_list)/processes))
    print("list ready")

    for submainwordlist in mainwordlistsplitted:
        print("sub")
        p = Process(target=do_job2, args=(seq_list, sub_seq_bank, data_list))
        p.Daemon = True
        p.start()
    for submainwordlist in mainwordlistsplitted:
        p.join()
  
    #----------------------------------------------------------------------------------
    print("done parallel")
    print(len(data_list))
    print("--- %s seconds ---" % (time.time() - start_time))   

    # single process --------------------------------------------------------------------
    print("done single")
    
    start_time = time.time()
    do_job2(seq_list, sub_seq_bank, data_list)   
    print("--- %s seconds ---" % (time.time() - start_time))   
    print(len(data_list))

    # # mul process --------------------------------------------------------------------
    # print("mul single")
    # data_list1 = []
    # start_time = time.time()
    # data_list1 = multiprocessing(do_job, seq_list, 2)
    # print("--- %s seconds ---" % (time.time() - start_time))   
    # print(len(data_list1[0]))

 
