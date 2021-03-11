#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import csv
import hashlib
from typing import ContextManager
import srt 
import pandas
import requests

import functools

from tqdm import tqdm
from pydub import AudioSegment

from datetime import datetime, timedelta
from pathlib import Path
from praatio import tgio

from .clean_transcript import clean_transcript


ALPHABET_FILE_PATH = "/DeepSpeech/bin/bangor_welsh/alphabet.txt"



def import_csv_textcorpus(csv_file_path, lm_data_root_dir):
    
    print ("Extracting texts from csv file: %s " % csv_file_path)
    if not os.path.isfile(csv_file_path):
        print ("Proceeding with missing file %s " % csv_file_path)

    Path(lm_data_root_dir).mkdir(parents=True, exist_ok=True)    
    corpus_file_path = os.path.join(lm_data_root_dir, "corpus.txt")

    df = pandas.read_csv(csv_file_path, encoding='utf-8', sep=',', header=0, dtype={'transcript':str})
    sentences = df['transcript']

    with open(corpus_file_path, 'w', encoding='utf-8') as corpus_file:
        for t in sentences:
            corpus_file.write(t + "\n")

    return clean_text_corpus(lm_data_root_dir)       


def clean_text_corpus(lm_data_root_dir):

    print ("Cleaning corpus files in %s " % lm_data_root_dir)
    
    source_text_file_path = os.path.join(lm_data_root_dir, "corpus.txt")
    output_text_file_path = os.path.join(lm_data_root_dir, "corpus.clean.txt")

    ooa_text_file_path = source_text_file_path.replace(".txt", ".ooa.txt")
    clean = clean_transcript(ALPHABET_FILE_PATH, ooa_text_file_path)
    
    with open(output_text_file_path, 'w', encoding='utf-8') as out_file:
        with open(source_text_file_path, 'r', encoding='utf-8') as in_file:
            for i, transcript in enumerate(tqdm(in_file)):
                cleaned, transcript = clean.clean(transcript)
                if cleaned:
                    out_file.write(transcript.lower() + "\n")

    return output_text_file_path


def get_macsen_textcorpus(url, lm_data_root_dir):

    target_dir = os.path.join(lm_data_root_dir, 'macsen')
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    json_data = json.loads(requests.get(url).text)
    with open(os.path.join(target_dir, "corpus.txt"), 'w', encoding='utf-8') as macsen_file_out: 
        for s in json_data["result"]:
            macsen_file_out.write(s[0] + "\n")

    return clean_text_corpus(target_dir)


def join_corpus_files(corpus_files, target_languagemodel_data_root_dir, joined_file_name):
    
    corpus_file_path = os.path.join(target_languagemodel_data_root_dir, joined_file_name)

    print ("Join corpus text files %s into %s" % (corpus_files, corpus_file_path) )

    with open(corpus_file_path, 'w', encoding='utf-8') as corpus_outfile:
        for fname in corpus_files:
            with open(fname, 'r', encoding='utf-8') as corpus_infile:
                for line in corpus_infile:
                    corpus_outfile.write(line)

    return clean_text_corpus(target_languagemodel_data_root_dir)