#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas
import functools

from pathlib import Path

from utils.imports import import_textgrid, import_srt
from utils.corpus import import_csv_textcorpus, join_corpus_files
from utils.kfold import create_kfolds

from argparse import ArgumentParser, RawTextHelpFormatter

DESCRIPTION = """
Paratoi adnoddau ychwanegol ar gyfer hyfforddi rhagor.

© Prifysgol Bangor University

"""
def create_kfolds_and_lm_data(bangor_data_root_dir, csv_file_path, target_kfolds_dir):
    create_kfolds(csv_file_path, target_kfolds_dir, 10)

    target_languagemodel_data_root_dir = os.path.join(target_kfolds_dir, "lm-data")
    Path(target_languagemodel_data_root_dir).mkdir(parents=True, exist_ok=True)

    base_text_corpus_file_path = os.path.join(bangor_data_root_dir, "lm-data", "corpus.clean.txt")
    for kfold_csv in os.listdir(target_kfolds_dir):
        if kfold_csv.startswith("train_"):
            corpus_files = []
            corpus_files.append(import_csv_textcorpus(os.path.join(target_kfolds_dir, kfold_csv), target_languagemodel_data_root_dir))
            corpus_files.append(base_text_corpus_file_path)
            corpus_file_path = join_corpus_files(corpus_files, target_languagemodel_data_root_dir, "corpus.%s.union.clean.txt" % kfold_csv.replace(".csv",""))
            print ("lm-data for kfold %s created in %s" % (kfold_csv, corpus_file_path))


def main(bangor_data_root_dir, finetune_data_file, target_finetuning_root_dir, **args):

    target_languagemodel_data_root_dir = os.path.join(target_finetuning_root_dir, "lm-data")
    Path(target_languagemodel_data_root_dir).mkdir(parents=True, exist_ok=True)

    target_csv_file_path = os.path.join(target_finetuning_root_dir, "deepspeech.csv")
    
    #
    transcribed_clips = []
    with open(finetune_data_file, 'r', encoding='utf-8') as finetune_files:
        for finetune_file_path in finetune_files:            
            if finetune_file_path.startswith("#"):
                continue

            finetune_file_path = finetune_file_path.rstrip()
            if not os.path.isfile(finetune_file_path):
                continue
            
            if finetune_file_path.endswith(".TextGrid"):
                transcribed_clips.append(import_textgrid(target_csv_file_path, finetune_file_path))
            elif finetune_file_path.endswith(".srt"):
                transcribed_clips.append(import_srt(target_csv_file_path, finetune_file_path))
    
    df_transcribed_clips = pandas.concat(transcribed_clips)
    df_transcribed_clips.to_csv(target_csv_file_path, index=False)
    
    # collect transcriptions into additions for fine tune training a language model
    base_text_corpus_file_path = os.path.join(bangor_data_root_dir, "lm-data", "corpus.clean.txt")
    corpus_files = []   
    corpus_files.append(import_csv_textcorpus(target_csv_file_path, target_languagemodel_data_root_dir))
    corpus_files.append(base_text_corpus_file_path)
    corpus_file_path = join_corpus_files(corpus_files, target_languagemodel_data_root_dir, "corpus.union.clean.txt")

    # create k-folds for determining new WER from fine tuned data. 
    create_kfolds_and_lm_data(bangor_data_root_dir, target_csv_file_path, os.path.join(target_finetuning_root_dir, "kfolds"))

    #
    print ("Import fine tuning data to %s finished." % (target_finetuning_root_dir))
    print ("Corpus for fine tuning language model is at %s" % corpus_file_path)


if __name__ == "__main__":
    
    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)

    parser.add_argument("--bangor_dir", dest="bangor_data_root_dir", default="/data/bangor")
    parser.add_argument("--target_dir", dest="target_finetuning_root_dir", help="target folder for all finetuning resources. should have an accompanying wav file (of the same name but with .wav extension)", default="/data/finetuning")
    parser.add_argument("--finetuning_data_file", dest="finetune_data_file", help="File containing paths (one per line) to srt and/or TextGrid files", required=True)
    # parser.add_argument("-c", dest="base_text_corpus_file_path", help=" file path to a text corpus that will be fined tuned e.g. OSCAR corpus", required=True)
    
    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(**vars(args))
