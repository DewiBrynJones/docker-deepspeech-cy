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
Llwytho i lawr set data amcan Macsen 

Mae angen rhoid lleoliad i ffeil alphabet.txt

© Prifysgol Bangor University

"""


def main(target_destination_root_dir, url_to_custom_macsen_skills_server, **args):

    #
    target_languagemodel_data_root_dir = os.path.join(target_destination_root_dir, "lm-data")
    Path(target_languagemodel_data_root_dir).mkdir(parents=True, exist_ok=True)
    
    # language model for Macsen downloaded from API..
    get_macsen_textcorpus(url_to_custom_macsen_skills_server, target_languagemodel_data_root_dir)
    print ("Custom Macsen text corpus ready at %s " % target_languagemodel_data_root_dir )


if __name__ == "__main__":
    
    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)    
    parser.add_argument("--output_dir", dest="target_destination_root_dir", default="/data/custom")
    parser.add_argument("--url", dest="url_to_custom_macsen_skills_server", help="custom macsen skills server e.g. https://mywebsite.com/assistant/get_all_sentences" required=True)
    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(**vars(args))
