#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import csv
import hashlib
from typing import ContextManager
import srt 
import pandas
import functools

from pydub import AudioSegment

from datetime import datetime, timedelta
from pathlib import Path

from praatio import tgio

from .clean_transcript import clean_transcript


ALPHABET_FILE_PATH = "/DeepSpeech/bin/bangor_welsh/alphabet.txt"

def get_directory_structure(rootdir):
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir, followlinks=True):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = functools.reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir

    return dir


def import_textgrid(target_csv_file, textfile):

    print ("Importing clips and transcripts from %s " % textfile)
    target_data_root_dir = Path(target_csv_file).parent
    
    target_clips_dir = os.path.join(target_data_root_dir, "clips")
    Path(target_clips_dir).mkdir(parents=True, exist_ok=True)

    df = pandas.DataFrame(columns=['wav_filename', 'wav_filesize', 'transcript'])

    textgrid_file_path = os.path.join(target_data_root_dir, textfile)
    soundfile = textgrid_file_path.replace(".TextGrid",".wav")

    audio_file = AudioSegment.from_wav(os.path.join(target_data_root_dir, soundfile))

    ooa_text_file_path = os.path.join(target_data_root_dir, 'deepspeech.ooa.txt')
    clean = clean_transcript(ALPHABET_FILE_PATH, ooa_text_file_path)

    tg = tgio.openTextgrid(textgrid_file_path)
    entryList = tg.tierDict["utterance"].entryList
    i=0
    for interval in entryList:
        text = interval.label        
        cleaned, transcript = clean.clean(text)
            
        if cleaned and len(transcript)>0:
            transcript = transcript.lower()
                
            start = float(interval.start) * 1000
            end = float(interval.end) * 1000

            #print (start, end, transcript)

            split_audio = audio_file[start:end]  
            hashId = hashlib.md5(transcript.encode('utf-8')).hexdigest()
            wav_segment_filepath = os.path.join(target_clips_dir, hashId + ".wav")   
            split_audio.export(wav_segment_filepath, format="wav")

            df.loc[i] = [wav_segment_filepath, os.path.getsize(wav_segment_filepath), transcript]
            i += 1

    return df


def import_srt(target_csv_file, srtfile):

    print ("Importing transcripts from srt file in %s " % srtfile)    
    target_data_root_dir = Path(target_csv_file).parent

    target_clips_dir = os.path.join(target_data_root_dir, "clips")
    Path(target_clips_dir).mkdir(parents=True, exist_ok=True)

    df = pandas.DataFrame(columns=['wav_filename', 'wav_filesize', 'transcript'])

    srt_file_path = os.path.join(target_data_root_dir, srtfile)
    soundfile = srt_file_path.replace(".srt",".wav")

    audio_file = AudioSegment.from_wav(os.path.join(target_data_root_dir, soundfile))

    ooa_text_file_path = os.path.join(target_data_root_dir, 'deepspeech.ooa.txt')
    clean = clean_transcript(ALPHABET_FILE_PATH, ooa_text_file_path)

    subs = list(srt.parse(open(srt_file_path, 'r', encoding='utf-8').read()))
    i = 0
    for s in subs:
        text = s.content 
        cleaned, transcript = clean.clean(text)
            
        if cleaned and len(transcript)>0:
            transcript = transcript.lower()

            start = float(s.start.total_seconds()) * 1000
            end = float(s.end.total_seconds()) * 1000

            #print (start, end, transcript)

            split_audio = audio_file[start:end]  
            hashId = hashlib.md5(transcript.encode('utf-8')).hexdigest()
            wav_segment_filepath = os.path.join(target_clips_dir, hashId + ".wav")   
            split_audio.export(wav_segment_filepath, format="wav")

            df.loc[i] = [wav_segment_filepath, os.path.getsize(wav_segment_filepath), transcript]
            i += 1                                    

    return df



def import_clips_dir(target_testset_dir, **args):

    print ("Importing clips dir in %s " % target_testset_dir)

    arddweud_root_dir = get_directory_structure(os.path.join(target_testset_dir, "clips"))
    
    csv_file_path = os.path.join(target_testset_dir, 'deepspeech.csv')
    print (csv_file_path)

    moz_fieldnames = ['wav_filename', 'wav_filesize', 'transcript']
    csv_file_out = csv.DictWriter(open(csv_file_path, 'w', encoding='utf-8'), fieldnames=moz_fieldnames)
    csv_file_out.writeheader()

    ooa_text_file_path = os.path.join(target_testset_dir, 'deepspeech.ooa.txt')
    clean = clean_transcript(ALPHABET_FILE_PATH, ooa_text_file_path)

    for filename in arddweud_root_dir["clips"]:
        if filename.endswith(".wav"):
            wavfilepath = os.path.join(target_testset_dir, "clips", filename)
            txtfilepath = wavfilepath.replace(".wav", ".txt")
            with open(txtfilepath, "r", encoding='utf-8') as txtfile:
                transcript = txtfile.read()
                cleaned, transcript = clean.clean(transcript)
                if cleaned:
                    transcript = transcript.lower()
                    if audio.downsample_wavfile(wavfilepath):                        
                        # print (wavfilepath)
                        csv_file_out.writerow({
                            'wav_filename':wavfilepath, 
                            'wav_filesize':os.path.getsize(wavfilepath), 
                            'transcript':transcript
                        })
    
    #return pandas.read_csv(csv_file_path, delimiter=',', encoding='utf-8')
    return csv_file_path
