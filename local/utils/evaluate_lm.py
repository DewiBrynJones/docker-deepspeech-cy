#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import kenlm

from argparse import ArgumentParser, RawTextHelpFormatter

DESCRIPTION = """

"""

def main(text_file_path, language_model_file_path, **args):
    model = kenlm.LanguageModel(language_model_file_path)
    print('{0}-gram model'.format(model.order))

    with open(text_file_path, 'r', encoding='utf-8') as in_text:
        for text in in_text:
            print ("\n\n{0} : {1}".format(text.rstrip(), model.score(text)))
            words = ['<s>'] + text.split() + ['</s>']
            for i, (prob, length, oov) in enumerate(model.full_scores(text)):
                print('{0} {1}: {2}'.format(prob, length, ' '.join(words[i+2-length:i+2])))
                if oov:
                    print('\t"{0}" is an OOV'.format(words[i+1]))
            for w in words:
                if not w in model:
                    print('"{0}" is an OOV'.format(w))


if __name__ == "__main__":
    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter) 
    parser.add_argument("-t", dest="text_file_path", required=True)
    parser.add_argument("-l", dest="language_model_file_path", required=True)
    
    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(**vars(args))
