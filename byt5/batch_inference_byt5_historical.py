restart = False # set to true to re-run everything, even if data is already there

# ------- on home -------

# # ----- just words --------
# output_dir = '/Users/jnaiman/Downloads/tmp/byt5_inline_cite_ref_small_words/' 
# # where to save everybody
# output_dir_inf = '/Users/jnaiman/Dropbox/wwt_image_extraction/OCRPostCorrection/inferences/historical/'
# snapshot = 'checkpoint-31000' # set to None to take last
# ender = '_small_words' # 100k for training, 5k val
# only_words = True

# -------- full --------
output_dir = '/Users/jnaiman/Downloads/tmp/byt5_inline_cite_ref_large/' 
# where to save everybody
output_dir_inf = '/Users/jnaiman/Dropbox/wwt_image_extraction/OCRPostCorrection/inferences/historical_full/'
snapshot = 'checkpoint-87000' # set to None to take last
ender = '_full_large' # 100k for training, 5k val
only_words = False

# also, test with a model that has been trained on *just* the historical dataset
# (this runs at the same time)
output_dir_hist = '/Users/jnaiman/Downloads/tmp/byt5_ocr_full_hal_historical/' 
# where to save everybody
snapshot_hist = 'checkpoint-160' # set to None to take last

aligned_dataset_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/OCRPostCorrection/alignments/'
historical_dataset_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/OCRPostCorrection/historical_docs/groundtruth/'
main_sanskrit_dir = '/Users/jnaiman/pe-ocr-sanskrit/' # should change this
library_dir = '../'


# which types of groundtruths are we wanting to use?
types = ['plain', 'author location']

imod = 10

nProcs = 6

special_char = ['\\%', '\\&']

skip_specials = True
verbose = True


# wait_timeout = 5.0 # timeout in minutes
# batch_size=100
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

# def fix_ocr(dfout_historical):
#     dfout_err_h = dfout_historical.copy()

#     dfmask = []

#     cer_orig = []; wer_orig = []
#     cer_corr = []; wer_corr = []
#     cer_corr_fix = []; wer_corr_fix = []
#     cer_fix = []; wer_fix = []
#     pdf_fix_out = []; ocr_fix_out = []
#     for i in range(len(dfout_historical)):

#         ocr = dfout_historical.iloc[i]['input_text']
#         pdf = dfout_historical.iloc[i]['target_text']
#         ocr_corr = dfout_historical.iloc[i]['predicted_text']

#         # also, do with just taking out locations of citations, refs and inlines:
#         # ------ 1. inline math --------
#         ind = 0
#         pdf_fix = ''
#         while ind < len(pdf):
#             if '$' in pdf[ind:]:
#                 i1 = pdf[ind:].index('$')
#                 i2 = pdf[ind+i1+1:].index('$')+i1+1+1
#                 # find match
#                 pdf_fix += pdf[ind:ind+i1]
#                 pdf_fix += '$'
#                 #import sys; sys.exit()
#                 ind += i2
#             else:
#                 pdf_fix += pdf[ind:]
#                 ind += len(pdf[ind:])

#         ind = 0
#         ocr_corr_fix = ''
#         while ind < len(ocr_corr):
#             if '$' in ocr_corr[ind:]:
#                 i1 = ocr_corr[ind:].index('$')
#                 nope = False
#                 try:
#                     i2 = ocr_corr[ind+i1+1:].index('$')+i1+1+1
#                 except:
#                     print('no matching $ in OCR corrected fix!')
#                     nope = True
#                 if not nope:
#                     # find match
#                     ocr_corr_fix += ocr_corr[ind:ind+i1]
#                     ocr_corr_fix += '$'
#                     #import sys; sys.exit()
#                     ind += i2
#                 else:
#                     ind += i1+1
#             else:
#                 ocr_corr_fix += ocr_corr[ind:]
#                 ind += len(ocr_corr[ind:])

#         # ------ 2. citations -----
#         ind = 0
#         pdf_fix2 = ''
#         while ind < len(pdf_fix):
#             if '\\cite' in pdf_fix[ind:]:
#                 ind1,ind2 = spc(pdf_fix[ind:],function='\\cite',
#                         dopen='{',dclose='}',
#                        error_out=True)
#                 pdf_fix2 += pdf_fix[ind:ind+ind1]
#                 pdf_fix2 += '`'
#                 ind += ind2
#             else:
#                 pdf_fix2 += pdf_fix[ind:]
#                 ind += len(pdf_fix[ind:])

#         ind = 0
#         ocr_corr_fix2 = ''
#         while ind < len(ocr_corr_fix):
#             if '\\cite' in ocr_corr_fix[ind:]:
#                 ind1,ind2 = spc(ocr_corr_fix[ind:],function='\\cite',
#                         dopen='{',dclose='}',
#                        error_out=True)
#                 ocr_corr_fix2 += ocr_corr_fix[ind:ind+ind1]
#                 ocr_corr_fix2 += '`'
#                 ind += ind2
#             else:
#                 ocr_corr_fix2 += ocr_corr_fix[ind:]
#                 ind += len(ocr_corr_fix[ind:])

#          # ------ 3. refs -----
#         ind = 0
#         pdf_fix3 = ''
#         while ind < len(pdf_fix2):
#             if '\\ref' in pdf_fix2[ind:]:
#                 ind1,ind2 = spc(pdf_fix2[ind:],function='\\ref',
#                         dopen='{',dclose='}',
#                        error_out=True)
#                 pdf_fix3 += pdf_fix2[ind:ind+ind1]
#                 pdf_fix3 += '*'
#                 ind += ind2
#             else:
#                 pdf_fix3 += pdf_fix2[ind:]
#                 ind += len(pdf_fix2[ind:])

#         ind = 0
#         ocr_corr_fix3 = ''
#         while ind < len(ocr_corr_fix2):
#             if '\\ref' in ocr_corr_fix2[ind:]:
#                 ind1,ind2 = spc(ocr_corr_fix2[ind:],function='\\ref',
#                         dopen='{',dclose='}',
#                        error_out=True)
#                 ocr_corr_fix3 += ocr_corr_fix2[ind:ind+ind1]
#                 ocr_corr_fix3 += '*'
#                 ind += ind2
#             else:
#                 ocr_corr_fix3 += ocr_corr_fix2[ind:]
#                 ind += len(ocr_corr_fix2[ind:])

#         #if i == 5: import sys; sys.exit()

#         # orig errors
#         cer_orig_here = fastwer.score_sent(ocr,pdf, 
#                                       char_level=True)
#         wer_orig_here = fastwer.score_sent(ocr,pdf, 
#                                       char_level=False)
#         # after correction
#         cer_corr_here = fastwer.score_sent(ocr_corr,pdf, 
#                                       char_level=True)
#         wer_corr_here = fastwer.score_sent(ocr_corr,pdf, 
#                                       char_level=False)

#         # after correction, with replacement
#         cer_corr_fix_here = fastwer.score_sent(ocr_corr_fix3,pdf_fix3, 
#                                       char_level=True)
#         wer_corr_fix_here = fastwer.score_sent(ocr_corr_fix3,pdf_fix3, 
#                                       char_level=False)

#         # before correction, with replacement
#         cer_fix_here = fastwer.score_sent(ocr,pdf_fix3, 
#                                       char_level=True)
#         wer_fix_here = fastwer.score_sent(ocr,pdf_fix3, 
#                                       char_level=False)


#         # ignore if we haven't checked for extra non-tracked things
#         pdf_check = pdf_fix3.replace('*','').replace('`','').replace('$','')
#         s = re.search(search_doc, pdf_check)
#         if s: # skip
#             dfmask.append(False)
#             continue
#         else:
#             dfmask.append(True)


#         pdf_fix_out.append(pdf_fix3)
#         ocr_fix_out.append(ocr_corr_fix3)


#         cer_orig.append(cer_orig_here)
#         wer_orig.append(wer_orig_here)
#         cer_corr.append(cer_corr_here)
#         wer_corr.append(wer_corr_here)
#         cer_corr_fix.append(cer_corr_fix_here)
#         wer_corr_fix.append(wer_corr_fix_here)
#         cer_fix.append(cer_fix_here)
#         wer_fix.append(wer_fix_here)

#     # recopy
#     dfout_err_h = dfout_historical.copy().loc[dfmask]

#     dfout_err_h['CER Orig'] = cer_orig
#     dfout_err_h['WER Orig'] = wer_orig
#     dfout_err_h['CER Corrected'] = cer_corr
#     dfout_err_h['WER Corrected'] = wer_corr
#     dfout_err_h['CER Fix Corrected'] = cer_corr_fix
#     dfout_err_h['WER Fix Corrected'] = wer_corr_fix
#     dfout_err_h['CER Fix'] = cer_fix
#     dfout_err_h['WER Fix'] = wer_fix
#     dfout_err_h['target_text_fixed'] = pdf_fix_out
#     dfout_err_h['predicted_text_fixed'] = ocr_fix_out
    
#     return dfout_err_h


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

        
# def store_file(fullpath, checkpoint, df_test, save_file=None, wait=2.0,
#               verbose=True,batch_size=100): # timeout in minutes
#     wait *= 60.0
#     ocr_pipeline = pipeline(
#         'text2text-generation',
#         model = fullpath,
#         tokenizer=tokenizer,
#     batch_size=batch_size)

#     #print('Model Loaded')
#     start = time.time()
#     #print('Time is ', start)
#     results = [] 
#     data = list(df_test.input_text.values)
#     err = False
#     try:
#         with timeout(seconds=int(wait)):
#             results = ocr_pipeline(data)

#         if verbose: print('Total time taken to process is ', round(time.time()-start,2), 'seconds')
#         pred_resultz = []
#         for i in list(range(len(results))):
#             for k,e in results[i].items():
#                 pred_resultz.append(e)

#         res = pd.DataFrame(zip(df_test.input_text.values,
#                                df_test.target_text.values,
#                                pred_resultz),columns = ['input_text','target_text','predicted_text'])

#     except:
#         print('Timeout it is then...')
#         res = pd.DataFrame({'input_text':[],'target_text':[],'predicted_text':[]})
#         err = True
        
#     if save_file is not None:
#         res.to_csv(save_file,index = False,sep=';')

#     return res, err

# # this just uses a model
# def store_file_model(model, df_test, save_file=None, wait=2.0,
#               verbose=True,batch_size=100): # timeout in minutes
#     wait *= 60.0
#     ocr_pipeline = pipeline(
#         'text2text-generation',
#         model = model,
#         tokenizer=tokenizer,batch_size=batch_size)

#     #print('Model Loaded')
#     start = time.time()
#     #print('Time is ', start)
#     results = [] 
#     data = list(df_test.input_text.values)
#     err = False
#     try:
#         with timeout(seconds=int(wait)):
#             results = ocr_pipeline(data)

#         if verbose: print('Total time taken to process is ', round(time.time()-start,2), 'seconds')
#         pred_resultz = []
#         for i in list(range(len(results))):
#             for k,e in results[i].items():
#                 pred_resultz.append(e)

#         res = pd.DataFrame(zip(df_test.input_text.values,
#                                df_test.target_text.values,
#                                pred_resultz),columns = ['input_text','target_text','predicted_text'])

#     except:
#         print('Timeout it is then...')
#         res = pd.DataFrame({'input_text':[],'target_text':[],'predicted_text':[]})
#         err = True
        
#     if save_file is not None:
#         res.to_csv(save_file,index = False,sep=';')

#     return res, err

# class GPReviewDataset(Dataset):
#     def __init__(self, Text, Label):
#         self.Text = Text
#         self.Label = Label
#         # self.tokenizer = tokenizer
#         # self.max_len = max_len
#     def __len__(self):
#         return len(self.Text)
#     def __getitem__(self, item):
#         Text = str(self.Text[item])
#         Label = self.Label[item]
#         inputs = tokenizer(Text, padding="max_length", truncation=True, max_length=512)
#         outputs = tokenizer(Label, padding="max_length", truncation=True, max_length=512)
#         return {
#           "input_ids":inputs.input_ids,
#           "attention_mask" : inputs.attention_mask,
#           "labels" : outputs.input_ids,
#           "decoder_attention_mask" : outputs.attention_mask,
#           # "labels" : lbz
#         }
    
    
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

# test_df = pd.read_csv(aligned_dataset_dir + test_latex)
# test_df = test_df.rename(columns={"sentences source": "input_text", 
#                         "sentences target": "target_text"})

# args_dict = {
#     #"model_name_or_path": 'google/byt5-small',
#     #"max_len": 4096,
#     #"max_length": 4096,
#     "output_dir": output_dir,
#     "overwrite_output_dir": True,
#     "per_device_train_batch_size": 4,
#     "per_device_eval_batch_size": 4,
#     "gradient_accumulation_steps": 4,
#     "learning_rate": 5e-4,
#     "warmup_steps": 250,
#     "logging_steps": 100,
#     "evaluation_strategy": "steps",
#     "eval_steps": 1000,
#     "num_train_epochs": 4,
#     "do_train": True,
#     "do_eval": True,
#     "fp16": False,
#     #"use_cache": False,
#     "max_steps": 100000,
#     'save_steps':1000,
#     'save_strategy':'steps',
#     # 'load_best_model_at_end': True,
#     # 'metric_for_best_model':'eval_loss',
#     # 'greater_is_better':False
# }

# parser = HfArgumentParser(
#         (TrainingArguments))
# training_args = parser.parse_dict(args_dict)
# # set_seed(training_args.seed)
# args = training_args[0]

# Load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "google/byt5-small",
    cache_dir=output_dir, 
    max_length=4096
)

model = T5ForConditionalGeneration.from_pretrained(
    "google/byt5-small",
    cache_dir=output_dir,
)

# overwriting the default max_length of 20 
tokenizer.model_max_length=4096
model.config.max_length=4096



# snapshots = glob(output_dir+'checkpoint*')
# order = []
# for s in snapshots:
#     order.append(s.split('-')[-1])
# argsort = np.argsort(np.array(order).astype('int'))
# snapshot = np.array(snapshots)[argsort][-1]
if snapshot == None:
    snapshots = glob(output_dir+'checkpoint*')
    order = []
    for s in snapshots:
        order.append(s.split('-')[-1])
    argsort = np.argsort(np.array(order).astype('int'))
    snapshot = np.array(snapshots)[argsort][-1]
else:
    snapshot = output_dir + snapshot

# historical trained only
if snapshot_hist == None:
    snapshots_hist = glob(output_dir_hist+'checkpoint*')
    order = []
    for s in snapshots_hist:
        order.append(s.split('-')[-1])
    argsort = np.argsort(np.array(order).astype('int'))
    snapshot_hist = np.array(snapshots_hist)[argsort][-1]
else:
    snapshot_hist = output_dir_hist + snapshot_hist


ckpoint = snapshot.split('-')[-1]
ckpoint_hist = snapshot_hist.split('-')[-1]

historical_gt = glob(historical_dataset_dir + '*pickle')

inds = np.arange(0,len(historical_gt)) 
# debug
#inds = inds[:12]
#imod = 2

# load all models, replace orig
model = T5ForConditionalGeneration.from_pretrained(snapshot)
model_hist = T5ForConditionalGeneration.from_pretrained(snapshot_hist)

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
            
    #check_fle = output_dir_inf + dir_test['filename'] + ender+'.csv'
    #import sys; sys.exit()
    
    filenames = []; pages = []; sents = []; 
    source = []; target = []
    types_here = []
    with open(historical_gt[indd],'rb') as f:
        dir_test = pickle.load(f)
        
    filenames.append(dir_test['filename'])
    pages.append(dir_test['page'])
    sents.append(dir_test['sent'])
    source.append(dir_test['source'])
    target.append(dir_test['target'])
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

        #df_historical_test_all = df_historical_test_all.rename(columns={"input_text_unclean": "input_text", 
        #                            "target_text_unclean": "target_text"})
        inds1 = np.where(target_types == 'W')
        if len(inds1) > 0: # have stuff
            x = "".join(np.array(list(df_historical_test['input_text_unclean'].values[0])))
            y = "".join(np.array(list(df_historical_test['target_text_unclean'].values[0])))
            # align x to y
            eops = Levenshtein.editops(x, y)
            xalign,yalign,xtypes,ytypes = align_texts_fast(x,y,eops,
                                                           pdf_types=target_types, 
                                                           ocr_types='W'*len(x))
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
        
#     dfout_here,err = store_file(snapshot, ckpoint, 
#                                 df_historical_test,
#                                verbose = False, wait=wait_timeout,
#                                batch_size=batch_size)
#     dfout_here_hist,err_hist = store_file(snapshot_hist, ckpoint_hist, 
#                                 df_historical_test,
#                                verbose = False, wait=wait_timeout,
#                                batch_size=batch_size)
    
#     # with orig model
#     dfout_here2,err2 = store_file_model(model, 
#                                 df_historical_test,
#                                verbose = False, wait=wait_timeout,
#                                batch_size=batch_size)

    err = False
    err_hist = False
    
    text = df_historical_test.iloc[0]['input_text']
    verbose_message = ''
    if verbose:
        verbose_message += 'i='+str(indd) + '\n'
        verbose_message += 'input text                : '+ text + '\n'
        verbose_message += 'target text               : '+ df_historical_test.iloc[0]['target_text'] + '\n'
    # base model
    
#     try:
#         with timeout(seconds=int(wait)):
#             results = ocr_pipeline(data)

#         if verbose: print('Total time taken to process is ', round(time.time()-start,2), 'seconds')
#         pred_resultz = []
#         for i in list(range(len(results))):
#             for k,e in results[i].items():
#                 pred_resultz.append(e)

#         res = pd.DataFrame(zip(df_test.input_text.values,
#                                df_test.target_text.values,
#                                pred_resultz),columns = ['input_text','target_text','predicted_text'])

#     except:
#         print('Timeout it is then...')
#         res = pd.DataFrame({'input_text':[],'target_text':[],'predicted_text':[]})
#         err = True
    

    try:
        with timeout(seconds=int(wait)):
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
        with timeout(seconds=int(wait)):
            inputs = tokenizer(text, padding="longest", return_tensors="pt")
            output = model_hist.generate(**inputs)
            output_text_hist = tokenizer.decode(output[0], 
                                           skip_special_tokens=skip_specials, 
                                           clean_up_tokenization_spaces=True)
            if verbose:
                verbose_message += 'historical model predicted: '+ output_text_hist + '\n'
    except:
        print(indd, ': error or timeout in historical model')
        err = True
        
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
         
    df2.to_csv(output_dir_inf + fname_out + ender+'.csv', index=False)
        
        


