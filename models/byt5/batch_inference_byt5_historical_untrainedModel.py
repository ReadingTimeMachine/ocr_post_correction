# this will test the effectiveness of using a plain, "untrained" model from huggingface

restart = False # set to true to re-run everything, even if data is already there

# -------- full --------
# output_dir = '/Users/jnaiman/Downloads/tmp/byt5_inline_cite_ref_large/' 
# # where to save everybody
# output_dir_inf = '/Users/jnaiman/Dropbox/wwt_image_extraction/OCRPostCorrection/inferences/historical_full/'
# snapshot = 'checkpoint-87000' # set to None to take last
# ender = '_full_large' # 100k for training, 5k val
# only_words = False

# # also, test with a model that has been trained on *just* the historical dataset
# # (this runs at the same time)
# output_dir_hist = '/Users/jnaiman/Downloads/tmp/byt5_ocr_full_hal_historical/' 
# # where to save everybody
# snapshot_hist = 'checkpoint-160' # set to None to take last

# where do alignments live
aligned_dataset_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/OCRPostCorrection/alignments/'
historical_dataset_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/OCRPostCorrection/historical_docs/groundtruth/'
main_sanskrit_dir = '/Users/jnaiman/pe-ocr-sanskrit/' # should change this

# where is utils stored?
library_dir = '../../'

output_dir_inf = '/Users/jnaiman/Dropbox/wwt_image_extraction/OCRPostCorrection/inferences/historical_untrained/'
ender = '_untrained' 
only_words = True # use only words

# what model to use for this training
output_dir = '/Users/jnaiman/Downloads/tmp/yelpfest_ocr/' # where to store model
model_name = 'yelpfeast/byt5-base-english-ocr-correction'


# which types of groundtruths are we wanting to use?
types = ['plain', 'author location']

imod = 10

nProcs = 6

special_char = ['\\%', '\\&']

skip_specials = True
verbose = True


wait_timeout = 2.0 # timeout in minutes
use_alt_sent = True # flag for which type of sentence to use as input
# -------------------------------------------------------------

from torch import cuda

use_cpu = False

device = 'cuda' if cuda.is_available() else 'cpu'
cuda.empty_cache()
if use_cpu:
    device = 'cpu'
#print(device)


from transformers import HfArgumentParser, TensorFlowBenchmark, TensorFlowBenchmarkArguments
import pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer
from transformers import EarlyStoppingCallback

# get metrics
from sys import path
path.append(main_sanskrit_dir)

from transformers import pipeline
from tqdm import tqdm
##import glob
from metrics import get_metric
import subprocess

import fastwer
from glob import glob
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time
import re
import signal
import os


from sys import path
path.append(library_dir)
import Levenshtein
from utils import split_function_with_delimiters_with_checks as spc
from utils import get_fill_in_types, align_texts_fast, fix_ocr

import yt
yt.enable_parallelism()

# -------------------------------------------------------------



class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

        

    
def add_formatted_columns(datain):
    source = []
    target = []
    source_aligned = []
    target_aligned = []
    for i in range(len(datain)):
        d = datain.iloc[i]
        s = np.array(list(d['aligned sentences source'])) # aligned source, with ^ symbols
        t = np.array(list(d['aligned sentences target'])) # aligned target, with @ symbols
        a = np.array(list(get_fill_in_types(d['aligned sentences target types'])))
        if len(s) == len(t):
            ss = "".join(s[np.where( (a == ' ') | (a == 'W') | (a == 'w'))[0]].tolist())
            tt = "".join(t[np.where( (a == ' ') | (a == 'W') | (a == 'w'))[0]].tolist())
        else:
            print('have issue, testing')
            if t[0] == ' ' and s[0] != ' ':
                t = np.array(list(d['aligned sentences target']))[1:] # aligned target, with @ symbols
                a = np.array(list(get_fill_in_types(d['aligned sentences target types'])))[1:]
                if len(s) == len(t):
                    ss = "".join(s[np.where( (a == ' ') | (a == 'W') | (a == 'w'))[0]].tolist())
                    tt = "".join(t[np.where( (a == ' ') | (a == 'W') | (a == 'w'))[0]].tolist())
                else:
                    print('not aligned, best guess')
                    import sys; sys.exit()

        source_aligned.append(ss.replace('^','@')) # align with original 
        target_aligned.append(tt)
        source.append(ss.replace('^',''))
        target.append(tt.replace('@',''))

    datain['words source aligned'] = source_aligned
    datain['words target aligned'] = target_aligned
    datain['words source'] = source
    datain['words target'] = target
    return datain

# -------------------------------------------------------------

wait_timeout *= 60.0 # to seconds



# Load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=output_dir, 
    max_length=4096
)

model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    cache_dir=output_dir,
)

# overwriting the default max_length of 20 
tokenizer.model_max_length=4096
model.config.max_length=4096


# if snapshot == None:
#     snapshots = glob(output_dir+'checkpoint*')
#     order = []
#     for s in snapshots:
#         order.append(s.split('-')[-1])
#     argsort = np.argsort(np.array(order).astype('int'))
#     snapshot = np.array(snapshots)[argsort][-1]
# else:
#     snapshot = output_dir + snapshot

# # historical trained only
# if snapshot_hist == None:
#     snapshots_hist = glob(output_dir_hist+'checkpoint*')
#     order = []
#     for s in snapshots_hist:
#         order.append(s.split('-')[-1])
#     argsort = np.argsort(np.array(order).astype('int'))
#     snapshot_hist = np.array(snapshots_hist)[argsort][-1]
# else:
#     snapshot_hist = output_dir_hist + snapshot_hist


# ckpoint = snapshot.split('-')[-1]
# ckpoint_hist = snapshot_hist.split('-')[-1]

historical_gt = glob(historical_dataset_dir + '*pickle')

inds = np.arange(0,len(historical_gt)) 
# debug
#inds = inds[:12]
#imod = 2

# # load all models, replace orig
# model = T5ForConditionalGeneration.from_pretrained(snapshot)
# model_hist = T5ForConditionalGeneration.from_pretrained(snapshot_hist)


start_time = time.time()

my_storage = {}
for sto, indd in yt.parallel_objects(inds, nProcs, storage=my_storage):
    fname_out = historical_gt[indd].split('.pickle')[0].split('/')[-1]
    # check if we want to restart or not
    goOn = True
    if not restart: # not restarting, then check
        if os.path.exists(output_dir_inf + fname_out + ender+'.csv'):
            goOn = False
    if not goOn:
        continue
                
    filenames = []; pages = []; sents = []; 
    source = []; target = []
    types_here = []
    with open(historical_gt[indd],'rb') as f:
        dir_test = pickle.load(f)
        
    filenames.append(dir_test['filename'])
    pages.append(dir_test['page'])
    sents.append(dir_test['sent'])
    source.append(dir_test['source'])
    if not use_alt_sent:
        target.append(dir_test['target'])
    else:
        target.append(dir_test['target_alt'])
    types_here.append(dir_test['type'])
    
    df_historical_test_all = pd.DataFrame({'input_text_unclean':source, 
                                           'target_text_unclean':target,
                                       'filename':filenames, 'page':pages,
                                       'sent num':sents, 'type':types_here})
        
    df_historical_test = df_historical_test_all.loc[df_historical_test_all['type'].isin(types)]
    if len(df_historical_test) == 0: # check if stuff in there
        print('pass on this type:', types_here[0])
        continue
        
    if not only_words: # nothing fancy
        df_historical_test = df_historical_test.rename(columns={"input_text_unclean": "input_text", 
                                "target_text_unclean": "target_text"})
    else: # take out all words
        # get only words if need be
        # ------ 1. inline math --------
        pdf = dir_test['target']
        target_types = np.repeat('W',len(dir_test['target'])) # all words to start
        ind = 0
        # mark inlines
        while ind < len(pdf):
            if '$' in pdf[ind:]:
                i1 = pdf[ind:].index('$')
                i2 = pdf[ind+i1+1:].index('$')+i1+1+1
                target_types[ind+i1:ind+i2] = 'I'
                ind += i2
            else:
                ind += len(pdf[ind:])

        # ------ 2. citations -----
        ind = 0
        while ind < len(pdf):
            if '\\cite' in pdf[ind:]:
                ind1,ind2 = spc(pdf[ind:],function='\\cite',
                        dopen='{',dclose='}',
                       error_out=True)
                if ind1 != -1 and ind2 != -1:
                    target_types[ind+ind1:ind+ind2] = 'C'            
                    ind += ind2
                else:
                    print('issue with matching {} in cite')
                    ind += 1
            else:
                ind += len(pdf[ind:])


         # ------ 3. refs -----
        ind = 0
        while ind < len(pdf):
            if '\\ref' in pdf[ind:]:
                ind1,ind2 = spc(pdf[ind:],function='\\ref',
                        dopen='{',dclose='}',
                       error_out=True)
                if ind1 != -1 and ind2 != -1:
                    target_types[ind+ind1:ind+ind2] = 'R'            
                    ind += ind2
                else:
                    print('issue with matching braket in ref!')
                    ind += 1
            else:
                ind += len(pdf[ind:])

                
        # special characters
        for sc in special_char:
            ind = 0
            while ind < len(pdf):
                if sc in pdf[ind:]:
                    # already tagged as something?
                    indexb = pdf[ind:].index(sc)
                    test_types = "".join(target_types[ind+indexb:ind+indexb+len(sc)])
                    if test_types == 'W'*len(test_types): # not already tagged
                        ind1 = indexb
                        ind2 = indexb+len(sc)
                        target_types[ind+ind1:ind+ind2] = 'S'            
                        ind += ind2
                    else:
                        ind += 1
                else:
                    ind += len(pdf[ind:])
        
        # look for any {} that haven't been tagged yet
        ind = 0
        while ind < len(pdf):
            if '{' in pdf[ind:]:
                # already tagged as something?
                indexb = pdf[ind:].index('{')
                if target_types[ind+indexb] == 'W': # not already tagged as something
                    #ind1,ind2 = spc(pdf[ind:],function='',
                    #        dopen='{',dclose='}',
                    #       error_out=True)
                    ind1 = indexb
                    ind2 = -1
                    try:
                        ind2 = pdf[ind+ind1:].index('}') + ind1 + 1
                    except:
                        print('no matching }')
                    if ind1 != -1 and ind2 != -1:
                        target_types[ind+ind1:ind+ind2] = 'B'            
                        ind += ind2
                    else:
                        print('issue with matching {} in other!')
                        ind += 1
                else:
                    ind += 1
            else:
                ind += len(pdf[ind:])
        
        # finally, anything else like a special char
        ind = 0
        while ind < len(pdf):
            if '\\' in pdf[ind:]:
                # already tagged as something?
                indexb = pdf[ind:].index('\\')
                if target_types[ind+indexb] == 'W': # not already tagged
                    ind1,ind2 = spc(pdf[ind:],function='\\',
                            dopen='{',dclose='}',
                           error_out=True)
                    if ind1 != -1 and ind2 != -1:
                        target_types[ind+ind1:ind+ind2] = 'O'            
                        ind += ind2
                    else:
                        print('issue with matching {} in other!')
                        ind += 1
                else:
                    ind += 1
            else:
                ind += len(pdf[ind:])

        inds1 = np.where(target_types == 'W')
        if len(inds1) > 0: # have stuff
            x = "".join(np.array(list(df_historical_test['input_text_unclean'].values[0])))
            if only_words:
                y = pdf
            else:
                y = "".join(np.array(list(df_historical_test['target_text_unclean'].values[0])))
            # align x to y
            eops = Levenshtein.editops(x, y)
            xalign,yalign,xtypes,ytypes = align_texts_fast(x,y,eops,
                                                           pdf_types=target_types, 
                                                           ocr_types='W'*len(x))
            # full_types_y = get_fill_in_types(ytypes)
            # full_types_x = get_fill_in_types(xtypes)
            # inds2_y = np.where(np.array(list(full_types_y)) == 'W')[0]
            # inds2_x = np.where(np.array(list(full_types_x)) == 'W')[0]
            full_types = get_fill_in_types(ytypes)
            inds2 = np.where(np.array(list(full_types)) == 'W')[0]
            df_historical_test['input_text'] = "".join(np.array(list(xalign))[inds2]).replace('^','')
            df_historical_test['target_text'] = "".join(np.array(list(yalign))[inds2]).replace('@','')
            print('ocr :', x)
            print('pdf :', y)
            print('input ocr    :', df_historical_test['input_text'].values[0])
            print('input target :', df_historical_test['target_text'].values[0])
        else:
            print('inds is zero')
            continue
        
    if indd%imod == 0:
        end_time = time.time()
        print('on', indd, 'of', len(inds),'in', end_time-start_time, 'seconds, or',
             (end_time-start_time)/60., 'minutes')
        start_time = time.time()

    err = False
    err_hist = False
    
    text = df_historical_test.iloc[0]['input_text']
    verbose_message = ''
    if verbose:
        verbose_message += 'i='+str(indd) + '\n'
        verbose_message += 'input text                : '+ text + '\n'
        verbose_message += 'target text               : '+ df_historical_test.iloc[0]['target_text'] + '\n'

    try:
        with timeout(seconds=int(wait_timeout)):
            inputs = tokenizer(text, padding="longest", return_tensors="pt")
            output = model.generate(**inputs)
            output_text = tokenizer.decode(output[0], 
                                           skip_special_tokens=skip_specials, 
                                           clean_up_tokenization_spaces=True)
            if verbose:
                verbose_message += 'default model predicted   : '+ output_text + '\n'
        
    except:
        print(indd, ': error or timeout in base model')
        err = True
        
    # historical model
    try:
        with timeout(seconds=int(wait_timeout)):
            inputs = tokenizer(text, padding="longest", return_tensors="pt")
            output = model_hist.generate(**inputs)
            output_text_hist = tokenizer.decode(output[0], 
                                           skip_special_tokens=skip_specials, 
                                           clean_up_tokenization_spaces=True)
            if verbose:
                verbose_message += 'historical model predicted: '+ output_text_hist + '\n'
    except:
        print(indd, ': error or timeout in historical model')
        err_hist = True
        
    if verbose:
        verbose_message += '\n'
        print(verbose_message)
        

    
    # check with all errors
    df2 = df_historical_test.copy()
    if not err: # no error in first
        df2['predicted_text'] = str(output_text)
    else:
        df2['predicted_text'] = np.nan
        
    if not err_hist:
        df2['predicted_text_histOnlyModel'] = str(output_text_hist)
    else:
        df2['predicted_text_histOnlyModel'] = np.nan
         
    #import sys; sys.exit()
    
    df2.to_csv(output_dir_inf + fname_out + ender+'.csv', index=False)
        
        


