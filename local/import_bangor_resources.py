#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import csv
import gzip
import pandas
import requests
import json

from enum import Enum

import gzip
import shutil
import functools

from urllib.parse import urlparse

from pathlib import Path
from git import Repo

import utils.kfold as kfold
import utils.audio as audio

from utils.clean_transcript import clean_transcript

from utils.imports import import_textgrid, import_clips_dir, get_directory_structure
from utils.corpus import clean_text_corpus, import_csv_textcorpus, join_corpus_files, get_macsen_textcorpus

from argparse import ArgumentParser, RawTextHelpFormatter

DESCRIPTION = """
Llwytho i lawr set data Macsen ar gyfer DeepSpeech o fewn yr ap Macsen 

Mae angen rhoid lleoliad i ffeil alphabet.txt

© Prifysgol Bangor University

"""

ALPHABET_FILE_PATH = "/DeepSpeech/bin/bangor_welsh/alphabet.txt"


TESTSET_URL = "https://git.techiaith.bangor.ac.uk/data-corpws-mewnol/corpws-profi-deepspeech"
MACSEN_TEXT_CORPUS_URL = "https://api.techiaith.org/assistant/get_all_sentences"


def clone_bangor_testset(target_testset_dir):
    Repo.clone_from(TESTSET_URL, target_testset_dir)


def get_commonvoice_textcorpus(commonvoice_validated_csv_file_path, lm_data_root_dir):
    target_dir = os.path.join(lm_data_root_dir, 'commonvoice')
    import_csv_textcorpus(commonvoice_validated_csv_file_path, target_dir)



def get_oscar_textcorpus(oscar_archive_file_path, lm_data_root_dir):

    print ("Extracting: %s" % oscar_archive_file_path)

    target_dir = os.path.join(lm_data_root_dir, 'oscar')
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    oscar_corpus_file_path = oscar_archive_file_path.replace('.gz','')
    corpus_file_path = os.path.join(target_dir, "corpus.txt")

    with gzip.open(oscar_archive_file_path, 'rb') as oscar_archive_file:
        with open(oscar_corpus_file_path, 'wb') as oscar_file:
            shutil.copyfileobj(oscar_archive_file, oscar_file)
    
    shutil.move(oscar_corpus_file_path, corpus_file_path)

    return clean_text_corpus(target_dir)



def import_macsen_testset(target_testset_dir, **args):

    print ("Importing Macsen test sets")

    macsen_root_dir = get_directory_structure(target_testset_dir)
    
    csv_file_path = os.path.join(target_testset_dir, 'deepspeech.csv')

    moz_fieldnames = ['wav_filename', 'wav_filesize', 'transcript']
    csv_file_out = csv.DictWriter(open(csv_file_path, 'w', encoding='utf-8'), fieldnames=moz_fieldnames)
    csv_file_out.writeheader()

    ooa_text_file_path = os.path.join(target_testset_dir, 'deepspeech.ooa.txt')
    clean = clean_transcript(ALPHABET_FILE_PATH, ooa_text_file_path)

    for userid in macsen_root_dir["macsen"]["clips"]:                
        for filename in macsen_root_dir["macsen"]["clips"][userid]:                        
            if filename.endswith(".wav"):
                wavfilepath = os.path.join(target_testset_dir, "clips", userid, filename)
                txtfilepath = wavfilepath.replace(".wav", ".txt")
                with open(txtfilepath, "r", encoding='utf-8') as txtfile:
                    transcript = txtfile.read()
                    cleaned, transcript = clean.clean(transcript)
                    if cleaned:
                        transcript = transcript.lower()
                        if audio.downsample_wavfile(wavfilepath):                        
                            #print (wavfilepath)
                            csv_file_out.writerow({
                                'wav_filename':wavfilepath, 
                                'wav_filesize':os.path.getsize(wavfilepath), 
                                'transcript':transcript
                            })
                                                
    kfold.create_kfolds(csv_file_path, target_testset_dir, 10)
   


def main(bangor_target_root_dir, oscar_archive_file_path, commonvoice_root_dir, **args):

    #
    target_testset_root_dir = os.path.join(bangor_target_root_dir, "testsets")
    
    target_languagemodel_data_root_dir = os.path.join(bangor_target_root_dir, "lm-data")
    Path(target_languagemodel_data_root_dir).mkdir(parents=True, exist_ok=True)

    # Bangor testset contains tests for Macsen (digital assistant) and more general purpose transcription    
    clone_bangor_testset(target_testset_root_dir)
    
    # import Macsen testset into our environment
    import_macsen_testset(os.path.join(target_testset_root_dir, "data", "macsen"))

    # import transcription resources from bangor testset
    df_csvs=[]
    
    csv_OpiwHxPPqRI_file_path = import_textgrid(os.path.join(target_testset_root_dir, "data", "trawsgrifio", "OpiwHxPPqRI"), "sain.wav", "sain.TextGrid")
    df_csvs.append(pandas.read_csv(csv_OpiwHxPPqRI_file_path, delimiter=',', encoding='utf-8'))

    csv_arddweud_200617_file_path=import_clips_dir(os.path.join(target_testset_root_dir, "data", "trawsgrifio", "arddweud_200617"))
    df_csvs.append(pandas.read_csv(csv_arddweud_200617_file_path, delimiter=',', encoding='utf-8'))


    ## merge sub-tests into one bigger test
    df_all_transcript_csvs = pandas.concat(df_csvs)
    df_all_transcript_csvs.to_csv(os.path.join(target_testset_root_dir, "data", "trawsgrifio", "deepspeech.csv"), encoding='utf-8', index=False)
    print ("Testsets ready at %s " % target_testset_root_dir)


    # Resources for building language models

    # language model for Macsen downloaded from API..
    get_macsen_textcorpus(MACSEN_TEXT_CORPUS_URL, target_languagemodel_data_root_dir)
    print ("Macen text corpus ready at %s " % target_languagemodel_data_root_dir )

    # language model for transcription made up of multiple text sources..
    corpus_files = []
    corpus_files.append(get_oscar_textcorpus(oscar_archive_file_path, target_languagemodel_data_root_dir))

    commonvoice_validated_csv_file_path = os.path.join(commonvoice_root_dir, "validated.tsv")
    corpus_files.append(get_commonvoice_textcorpus(commonvoice_validated_csv_file_path, target_languagemodel_data_root_dir))

    corpus_file_path = join_corpus_files(corpus_files, target_languagemodel_data_root_dir)
    print ("Transcription text corpus ready at %s " % corpus_file_path )




if __name__ == "__main__":
    
    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)

    parser.add_argument("--bangor_dir", dest="bangor_target_root_dir", default="/data/bangor")
    parser.add_argument("--oscar_archive", dest="oscar_archive_file_path", required=True)
    parser.add_argument("--cv_dir", dest="commonvoice_root_dir", required=True)

    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(**vars(args))
