#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import tarfile
import pandas
import csv
import shlex
import shutil
import subprocess
import glob

from pathlib import Path
from utils.clean_transcript import clean_transcript
from argparse import ArgumentParser, RawTextHelpFormatter

DESCRIPTION = """

"""

ALPHABET_FILE_PATH = "/DeepSpeech/bin/bangor_welsh/alphabet.txt"


def extract(source_tar_gz, target_dir):

    print ("Extracting: %s" % source_tar_gz)

    tar = tarfile.open(source_tar_gz, "r:gz")
    tar.extractall(target_dir)
    tar.close()
    
    # files may exist in levels of subdirectories. Need to move them
    # up to the target_dir
    extracted_clips_path = glob.glob(os.path.join(target_dir,"**","clips"), recursive=True)
    if len(extracted_clips_path) > 0:
        extracted_clips_parent_path = str(Path(extracted_clips_path[0]).parent)
        for file_path in glob.glob(extracted_clips_parent_path + "/*"):
            print ("Moving from %s to %s " % (file_path, target_dir)) 
            shutil.move(file_path, target_dir)



def panda_group(df, column, destination_file_path):

    df_grp_client = df.groupby(column).size().to_frame('count').reset_index()
    df_grp_client = df_grp_client.sort_values("count", ascending=False)
    df_grp_client.to_csv(destination_file_path)
    


def analyze_tsvs(cv_root_dir):
    #client_id	path	sentence	up_votes	down_votes	age	gender	accent	locale	segment
    tsv_files = Path(cv_root_dir).glob("*.tsv")
    for tsv_file_path in tsv_files:
        
        print ("Analyzing %s " % tsv_file_path)

        if 'reported.tsv' in str(tsv_file_path):
            continue
        
        df = pandas.read_csv(tsv_file_path, encoding='utf-8', sep='\t', header=0, dtype={'gender':str})

        panda_group(df, 'client_id', str(tsv_file_path).replace(".tsv",".counts.client_id.txt"))
        panda_group(df, 'sentence', str(tsv_file_path).replace(".tsv",".counts.sentence.txt"))
        
        panda_group(df, 'age', str(tsv_file_path).replace(".tsv",".counts.age.txt"))
        panda_group(df, 'gender', str(tsv_file_path).replace(".tsv",".counts.gender.txt"))

        # analyze clients by age and gender....    


    
def main(cv_archive_file_path, cv_root_dir, **args):

    extract(cv_archive_file_path, cv_root_dir)
    
    #    
    analyze_tsvs(cv_root_dir)

    #
    print ("Preparing for DeepSpeech with import_cv2.py")
    cmd = "python3 /DeepSpeech/bin/import_cv2.py %s --validate_label_locale /DeepSpeech/bin/bangor_welsh/utils/validate_label_locale.py" % (cv_root_dir)

    import_process = subprocess.Popen(shlex.split(cmd))
    import_process.wait()



if __name__ == "__main__": 

    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter) 

    parser.add_argument("--archive", dest="cv_archive_file_path", required=True, help="path to downloaded tar.gz containing speech corpus in CommonVoice v2.0 format")
    parser.add_argument("--target_dir", dest="cv_root_dir", required=True, help="target directory for extracted archive, also root directory for training data")
   
    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(**vars(args))
