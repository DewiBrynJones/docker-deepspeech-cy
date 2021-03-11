#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
import pandas as pd

from pathlib import Path

from sklearn.model_selection import KFold
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

import argparse

def create_kfolds(csvfile, dest_dir, k):
    
    print ("Splitting %s into kfolds" % csvfile)

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=2)

    try:
        speech_corpus_df = pd.read_csv(csvfile, encoding='utf-8', delimiter=',')
    except pd.errors.EmptyDataError:
        print ("Failed to split %s into kfolds" % csvfile)
        return

    fold_num = 0
    for train_indexes, test_indexes in kf.split(speech_corpus_df):
        fold_num+=1

        train_file = os.path.join(dest_dir, 'train_' + str(fold_num) + '.csv')
        train_df = pd.DataFrame(speech_corpus_df.iloc[train_indexes])
        train_df.to_csv(train_file, index=False, sep=',', encoding='utf-8')
        
        test_file = os.path.join(dest_dir, 'test_' + str(fold_num) + '.csv')
        test_df = pd.DataFrame(speech_corpus_df.iloc[test_indexes])
        test_df.to_csv(test_file, index=False, sep=',', encoding='utf-8')

        #print ('train: %s, test %s' % (train_indexes, test_indexes))



def main(csvfile, dest_dir, k, **args):
    create_kfolds(csvfile, dest_dir, k)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='k-fold')

    parser.add_argument('--k', 
                        dest="k",
                        default=10,
                        help='k')

    parser.add_argument('--csv', 
                        dest='csvfile',
                        required=True,
                        help='DeepSpeech CSV file.')

    parser.add_argument('--dest_dir',
                        dest='dest_dir',
                        required=True,
                        help='dest dir')

    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(**vars(args))

