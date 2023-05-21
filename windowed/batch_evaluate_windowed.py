# -------- Local -------
output_folder = '/Users/jnaiman/Downloads/tmp/ocrpost/data/morgan/' # local
model_save_dir = '/Users/jnaiman/Downloads/tmp/ocrpost/data/morgan/models/' # local
file_dir = '/Users/jnaiman/Dropbox/wwt_image_extraction/OCRPostCorrection/alignments/'
lib_dir = '../scienceDigitization/text_mining_ocr_and_pdf/ocr_spell_check/'
post_correction_repo_dir = '../'


# -------- HAL -----------
# output_folder = '/home/jnaiman/data/morgan/' # HAL
# model_save_dir = '/home/jnaiman/data/morgan/models/' # HAL
# eval_save_dir = '/home/jnaiman/data/morgan/eval/' # HAL
# file_dir = '/home/jnaiman/wwt_image_extraction/alignments/'
# lib_dir = '/home/jnaiman/libraries/'
# post_correction_repo_dir = '/home/jnaiman/repos/'

# ---------------------------

ender = '_small_words_pageLevel' # small has 100,000 for training, 5000 for dev
# model save dir
model_file = 'new_torch_file_new_pages_small_words_pageLevel.pt'
test_file = output_folder + "data/test_file"+ender+'.csv'


# its not 100% clear if we need this... setting a flag, but looks like we DO need it for memory issues
use_train_dev_size = True
train_size = 1000000
dev_size = 1000000
window_length = 100

do_full_eval = False

window_type = 'n-grams'
window_size = 20
decoding = 'beam'
weighting = 'uniform'
ntest = 100


##################################################################
import time
import pickle
import os
from tqdm.notebook import tqdm
from pathlib import Path
import pandas as pd
from nltk.lm import Vocabulary
import pickle
import sys
import torch
import importlib
from timeit import default_timer as t
sys.path.append(post_correction_repo_dir)
#from metrics import levenshtein
#import post_ocr_correction # https://github.com/jarobyte91/post_ocr_correction/tree/master
from pytorch_beam_search import seq2seq
import pandas as pd
#!pip install Levenshtein
from Levenshtein import distance
import Levenshtein 

import re
from tqdm.auto import tqdm

import numpy as np

from sys import path
path.append(lib_dir)
from importlib import reload
if 'home' in lib_dir: # on HAL
    import utils_ocr_mini
    reload(utils_ocr_mini)
    from utils_ocr_mini import align_texts_fast, get_fill_in_types
else: # on home
    import utils
    reload(utils)
    from utils import align_texts_fast, get_fill_in_types  
    
from pytorch_beam_search.seq2seq import Transformer, beam_search

import torch
import torch.utils.data as tud
import torch.nn as nn

######################################################K
class Seq2Seq(nn.Module): 
    """
    A generic sequence-to-sequence model. All other sequence-to-sequence models should extend this class 
    with a __init__ and forward methods, in the same way as in normal PyTorch.
    """
    def print_architecture(self):
        """
        Displays the information about the model in standard output. 
        """
        for k in self.architecture.keys():
            print(f"{k.replace('_', ' ').capitalize()}: {self.architecture[k]}")
        print(f"Trainable parameters: {sum([p.numel() for p in self.parameters()]):,}")
        print()

    def fit(self,#train_loader, 
            X_train, 
            Y_train, 
            X_dev = None, 
            Y_dev = None, 
            batch_size = 100, 
            epochs = 5, 
            learning_rate = 10**-4, 
            weight_decay = 0, 
            progress_bar = 0, 
            save_path = None):
        print("fit begins")
        best_dev_loss=float('inf')
        best_epoch=float('inf')
        """
        A generic training method with Adam and Cross Entropy.

        Parameters
        ----------    
        X_train: LongTensor of shape (train_examples, train_input_length)
            The input sequences of the training set.
            
        Y_train: LongTensor of shape (train_examples, train_output_length)
            The output sequences of the training set.
            
        X_dev: LongTensor of shape (dev_examples, dev_input_length), optional
            The input sequences for the development set.
            
        Y_train: LongTensor of shape (dev_examples, dev_output_length), optional
            The output sequences for the development set.
            
        batch_size: int
            The number of examples to process in each batch.

        epochs: int
            The number of epochs of the training process.
            
        learning_rate: float
            The learning rate to use with Adam in the training process. 
            
        weight_decay: float
            The weight_decay parameter of Adam (L2 penalty), useful for regularizing models. For a deeper 
            documentation, go to https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam            

        progress_bar: int
            Shows a tqdm progress bar, useful for tracking progress with large tensors.
            If equal to 0, no progress bar is shown. 
            If equal to 1, shows a bar with one step for every epoch.
            If equal to 2, shows the bar when equal to 1 and also shows a bar with one step per batch for every epoch.
            If equal to 3, shows the bars when equal to 2 and also shows a bar to track the progress of the evaluation
            in the development set.
            
        save_path: string, optional
            Path to save the .pt file containing the model parameters when the training ends.

        Returns
        -------
        performance: Pandas DataFrame
            DataFrame with the following columns: epoch, train_loss, train_error_rate, (optionally dev_loss and 
            dev_error_rate), minutes, learning_rate, weight_decay, model, encoder_embedding_dimension, 
            decoder_embedding_dimension, encoder_hidden_units, encoder_layers, decoder_hidden_units, decoder_layers, 
            dropout, parameters and one row for each of the epochs, containing information about the training process.
        """
        assert X_train.shape[0] == Y_train.shape[0]
        assert (X_dev is None and Y_dev is None) or (X_dev is not None and Y_dev is not None) 
        if (X_dev is not None and Y_dev is not None):
            assert X_dev.shape[0] == Y_dev.shape[0]
            dev = True
        else:
            dev = False
            

        train_dataset = tud.TensorDataset(X_train, Y_train)
        train_loader = tud.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)# make own class data loader to read in batches at a time
#         class YourDataset(torch.utils.data.Dataset):
#             def __init__(self) -> None:
#                 self.source = torch.load(scratch + "data/train_source_new.pt")[:train_size].to(device)
#                 self.target = torch.load(scratch + "data/train_target_new.pt")[:train_size].to(device)
                
#                 source_files = glob.glob(scratch + "data/train_source_new0*.pt")
# #                 target_files = glob.glob(scratch + "data/train_target_new0*.pt")
# #                 source_files.sort()
# #                 target_files.sort()
# #                 self.source_files = source_files
# #                 self.target_files = target_files
# #                 self.file_number = 0
# #                 self.source = torch.load(source_files[self.file_number])[:train_size].to(device)
# #                 self.target = torch.load(target_files[self.file_number])[:train_size].to(device)
#                 self.file_length = len(self.source)
                
#             def __getitem__(self, idx) -> torch.Tensor:
#                 if idx < self.file_length:
#                     # load one sample by index, e.g like this:
#                     source_sample = self.source[idx]
#                     target_sample = self.target[idx]
#                 else:
#                     self.file_number += 1
#                     self.source = torch.load(source_files[self.file_number])[:train_size].to(device)
#                     self.target = torch.load(target_files[self.file_number])[:train_size].to(device)
#                     self.file_length = len(self.source)
#                     source_sample = self.source[idx]
#                     target_sample = self.target[idx]
            
                    
#                     #pd.len() 

#                 # do some preprocessing, convert to tensor and what not

#                 return source_sample, target_sample

#             def __len__(self):
#                 return len(self.source)
#         creating dataloader
#         yourDataset = YourDataset()
#         train_loader = torch.utils.data.DataLoader(yourDataset, batch_size= batch_size, num_workers=0, shuffle=True)
    
        criterion = nn.CrossEntropyLoss(ignore_index = 0)
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate, weight_decay = weight_decay)
        performance = []
        start = timer()
        epochs_iterator = range(1, epochs + 1)
        if progress_bar > 0:
            epochs_iterator = tqdm(epochs_iterator)
            print("Training started")
        print("X_train.shape:", X_train.shape)
        print("Y_train.shape:", Y_train.shape)
        if dev:
            print("X_dev.shape:", X_dev.shape)
            print("Y_dev.shape:", Y_dev.shape)
        print(f"Epochs: {epochs:,}\nLearning rate: {learning_rate}\nWeight decay: {weight_decay}")
        header_1 = "Epoch | Train                "
        header_2 = "      | Loss     | Error Rate"
        rule = "-" * 29
        if dev:
            header_1 += " | Development          "
            header_2 += " | Loss     | Error Rate"
            rule += "-" * 24
        header_1 += " | Minutes"
        header_2 += " |"
        rule += "-" * 10
        print(header_1, header_2, rule, sep = "\n")
        for e in epochs_iterator:
            self.train()
            losses = []
            errors = []
            sizes = []
            train_iterator = train_loader
            if progress_bar > 1:
                train_iterator = tqdm(train_iterator)
            for x, y in train_iterator:
                # compute loss and backpropagate
                probabilities = self.forward(x, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]
                loss = criterion(probabilities, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # compute accuracy
                predictions = probabilities.argmax(1)
                batch_errors = (predictions != y)
                # append the results
                losses.append(loss.item())
                errors.append(batch_errors.sum().item())
                sizes.append(batch_errors.numel())
            train_loss = sum(losses) / len(losses)
            train_error_rate = 100 * sum(errors) / sum(sizes)
            t = (timer() - start) / 60
            status_string = f"{e:>5} | {train_loss:>8.4f} | {train_error_rate:>10.3f}"
            status = {"epoch":e,
                      "train_loss": train_loss,
                      "train_error_rate": train_error_rate}
            if dev:
                dev_loss, dev_error_rate = self.evaluate(X_dev, 
                                                         Y_dev, 
                                                         batch_size = batch_size, 
                                                         progress_bar = progress_bar > 2, 
                                                         criterion = criterion)
                status_string += f" | {dev_loss:>8.4f} | {dev_error_rate:>10.3f}"
                status.update({"dev_loss": dev_loss, "dev_error_rate": dev_error_rate})
            status.update({"training_minutes": t,
                           "learning_rate": learning_rate,
                           "weight_decay": weight_decay})
            performance.append(status)
            if save_path is not None: 
                print("dev =", dev)
                print("e =", e)
                print("dev loss =", dev_loss)
                print("best dev loss =", best_dev_loss)
                #if (not dev) or (e < 2) or (dev_loss < min([p["dev_loss"] for p in performance[:-1]])):
                if (not dev) or (e < 2) or (dev_loss < best_dev_loss):
                    torch.save(self.state_dict(), save_path)
                    print(status)
                    best_dev_loss = dev_loss
                    print("save path =", save_path)
            status_string += f" | {t:>7.1f}"
            print(status_string)
        print()
        return pd.concat((pd.DataFrame(performance), 
                          pd.DataFrame([self.architecture for i in performance])), axis = 1)\
               .drop(columns = ["source_index", "target_index"])
    
            
    def evaluate(self, 
                 X, 
                 Y, 
                 criterion = nn.CrossEntropyLoss(), 
                 batch_size = 128, 
                 progress_bar = False):
        """
        Evaluates the model on a dataset.
        
        Parameters
        ----------
        X: LongTensor of shape (examples, input_length)
            The input sequences of the dataset.
            
        Y: LongTensor of shape (examples, output_length)
            The output sequences of the dataset.
            
        criterion: PyTorch module
            The loss function to evalue the model on the dataset, has to be able to compare self.forward(X, Y) and Y
            to produce a real number.
            
        batch_size: int
            The batch size of the evaluation loop.
            
        progress_bar: bool
            Shows a tqdm progress bar, useful for tracking progress with large tensors.
            
        Returns
        -------
        loss: float
            The average of criterion across the whole dataset.
            
        error_rate: float
            The step-by-step accuracy of the model across the whole dataset. Useful as a sanity check, as it should
            go to zero as the loss goes to zero.
            
        """
        dataset = tud.TensorDataset(X, Y)
        loader = tud.DataLoader(dataset, batch_size = batch_size)
        self.eval()
        losses = []
        errors = []
        sizes = []
        with torch.no_grad():
            iterator = iter(loader)
            if progress_bar:
                iterator = tqdm(iterator)
            for batch in iterator:
                x, y = batch
                # compute loss
                probabilities = self.forward(x, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]
                loss = criterion(probabilities, y)
                # compute accuracy
                predictions = probabilities.argmax(1)
                batch_errors = (predictions != y)
                # append the results
                losses.append(loss.item())
                errors.append(batch_errors.sum().item())
                sizes.append(batch_errors.numel())
            loss = sum(losses) / len(losses)
            error_rate = 100 * sum(errors) / sum(sizes)
        return loss, error_rate 
    
class LSTM(Seq2Seq):
    def __init__(self, 
                 source_index, 
                 target_index, 
                 encoder_embedding_dimension = 32,
                 decoder_embedding_dimension = 32,
                 encoder_hidden_units = 128, 
                 encoder_layers = 2,
                 decoder_hidden_units = 128,
                 decoder_layers = 2,
                 dropout = 0.0):
        """
        A standard Seq2Seq LSTM model as in 'Learning Phrase Representations using RNN Encoder-Decoder 
        for Statistical Machine Translation' by Cho et al. (2014). 
        
        Parameters
        ----------
        in_vocabulary: dictionary
            Vocabulary with the index:token pairs for the inputs of the model.
            
        out_vocabulary: dictionary
            Vocabulary with the token:index pairs for the outputs of the model.
            
        encoder_embedding_dimension: int
            Dimension of the embeddings to feed into the encoder.
            
        decoder_embedding_dimension: int
            Dimension of the embeddings to feed into the decoder.
            
        encoder_hidden_units: int
            Hidden size of the encoder.
            
        encoder_layers: int
            Hidden layers of the encoder.
            
        decoder_hidden_units: int
            Hidden units of the decoder.
            
        decoder_layers: int
            Hidden layers of the decoder.
            
        dropout: float between 0.0 and 1.0
            Dropout rate to apply to whole model.
        """
        self.source_index = source_index
        self.target_index = target_index
        super().__init__()
        self.source_embeddings = nn.Embedding(len(source_index), encoder_embedding_dimension)
        self.target_embeddings = nn.Embedding(len(target_index), decoder_embedding_dimension)
        self.encoder_rnn = nn.LSTM(input_size = encoder_embedding_dimension, 
                                   hidden_size = encoder_hidden_units, 
                                   num_layers = encoder_layers,
                                   dropout = dropout)
        self.decoder_rnn = nn.LSTM(input_size = encoder_layers * encoder_hidden_units + decoder_embedding_dimension, 
                                   hidden_size = decoder_hidden_units, 
                                   num_layers = decoder_layers,
                                   dropout = dropout)
        self.output_layer = nn.Linear(decoder_hidden_units, len(target_index))
        self.architecture = dict(model = "Seq2Seq LSTM",
                                 source_index = source_index, 
                                 target_index = target_index, 
                                 encoder_embedding_dimension = encoder_embedding_dimension,
                                 decoder_embedding_dimension = decoder_embedding_dimension,
                                 encoder_hidden_units = encoder_hidden_units, 
                                 encoder_layers = encoder_layers,
                                 decoder_hidden_units = decoder_hidden_units,
                                 decoder_layers = decoder_layers,
                                 dropout = dropout)
        self.print_architecture()
        
    def forward(self, X, Y):
        """
        Forward method of the model.
        
        Parameters
        ----------
        X: LongTensor of shape (batch_size, input_length)
            Tensor of integers containing the inputs for the model.
            
        Y: LongTensor of shape (batch_size, output_length)
            Tensor of integers containing the output produced so far.
            
        Returns
        -------
        output: FloatTensor of shape (batch_size, output_length, len(out_vocabulary))
            Tensor of floats containing the inputs for the final Softmax layer (usually integrated into the loss function).
        """
        X = self.source_embeddings(X.T)
        encoder, (encoder_last_hidden, encoder_last_memory) = self.encoder_rnn(X)
        encoder_last_hidden = encoder_last_hidden.transpose(0, 1).flatten(start_dim = 1)
        encoder_last_hidden = encoder_last_hidden.repeat((Y.shape[1], 1, 1))
        Y = self.target_embeddings(Y.T)
        Y = torch.cat((Y, encoder_last_hidden), axis = -1)
        decoder, (decoder_last_hidden, decoder_last_memory) = self.decoder_rnn(Y)
        output = self.output_layer(decoder.transpose(0, 1))
        return output        
    
    
class ReversingLSTM(Seq2Seq):
    def __init__(self, 
                 source_index, 
                 target_index, 
                 encoder_embedding_dimension = 32,
                 decoder_embedding_dimension = 32,
                 encoder_hidden_units = 128, 
                 encoder_layers = 2,
                 decoder_hidden_units = 128,
                 decoder_layers = 2,
                 dropout = 0.0):
        """
        A standard Seq2Seq LSTM model that reverses the order of the input as in 
        'Sequence to sequence learning with Neural Networks' by Sutskever et al. (2014). 
        
        Parameters
        ----------
        in_vocabulary: dictionary
            Vocabulary with the index:token pairs for the inputs of the model.
            
        out_vocabulary: dictionary
            Vocabulary with the token:index pairs for the outputs of the model.
            
        encoder_embedding_dimension: int
            Dimension of the embeddings to feed into the encoder.
            
        decoder_embedding_dimension: int
            Dimension of the embeddings to feed into the decoder.
            
        encoder_hidden_units: int
            Hidden size of the encoder.
            
        encoder_layers: int
            Hidden layers of the encoder.
            
        decoder_hidden_units: int
            Hidden units of the decoder.
            
        decoder_layers: int
            Hidden layers of the decoder.
            
        dropout: float between 0.0 and 1.0
            Dropout rate to apply to whole model.
        """
        super().__init__()
        self.source_index = source_index
        self.target_index = target_index
        self.source_embeddings = nn.Embedding(len(source_index), encoder_embedding_dimension)
        self.target_embeddings = nn.Embedding(len(target_index), decoder_embedding_dimension)
        self.encoder_rnn = nn.LSTM(input_size = encoder_embedding_dimension, 
                                   hidden_size = encoder_hidden_units, 
                                   num_layers = encoder_layers,
                                   dropout = dropout)
        self.decoder_rnn = nn.LSTM(input_size = decoder_embedding_dimension, 
                                   hidden_size = decoder_hidden_units, 
                                   num_layers = decoder_layers,
                                   dropout = dropout)
        self.output_layer = nn.Linear(decoder_hidden_units, len(target_index))
        self.enc2dec = nn.Linear(encoder_hidden_units * encoder_layers, decoder_hidden_units * decoder_layers)
        self.architecture = dict(model = "Seq2Seq Reversing LSTM",
                                 source_index = source_index, 
                                 target_index = target_index, 
                                 encoder_embedding_dimension = encoder_embedding_dimension,
                                 decoder_embedding_dimension = decoder_embedding_dimension,
                                 encoder_hidden_units = encoder_hidden_units, 
                                 encoder_layers = encoder_layers,
                                 decoder_hidden_units = decoder_hidden_units,
                                 decoder_layers = decoder_layers,
                                 dropout = dropout)
        self.print_architecture()
        
    def forward(self, X, Y):
        """
        Forward method of the model.
        
        Parameters
        ----------
        X: LongTensor of shape (batch_size, input_length)
            Tensor of integers containing the inputs for the model.
            
        Y: LongTensor of shape (batch_size, output_length)
            Tensor of integers containing the output produced so far.
            
        Returns
        -------
        output: FloatTensor of shape (batch_size, output_length, len(out_vocabulary))
            Tensor of floats containing the inputs for the final Softmax layer (usually integrated into the loss function).
        """
        X = self.source_embeddings(torch.flip(X.T, dims = (1, )))
        encoder, (encoder_last_hidden, encoder_last_memory) = self.encoder_rnn(X)
        encoder_last_hidden = encoder_last_hidden.transpose(0, 1).flatten(start_dim = 1)
        enc2dec = self.enc2dec(encoder_last_hidden)\
        .reshape(-1, self.decoder_rnn.num_layers, self.decoder_rnn.hidden_size)\
        .transpose(0, 1)\
        .contiguous()
        Y = self.target_embeddings(Y.T)
        decoder, (decoder_last_hidden, decoder_last_memory) = self.decoder_rnn(Y, (enc2dec, torch.zeros_like(enc2dec)))
        output = self.output_layer(decoder.transpose(0, 1))
        return output
    
    
class Transformer(Seq2Seq):
    def __init__(self, 
                 source_index, 
                 target_index,
                 max_sequence_length = 32,
                 embedding_dimension = 32,
                 feedforward_dimension = 128,
                 encoder_layers = 2,
                 decoder_layers = 2,
                 attention_heads = 2,
                 activation = "relu",
                 dropout = 0.0):
        """
        The standard PyTorch implementation of a Transformer model.
        
        Parameters
        ----------
        in_vocabulary: dictionary
            Vocabulary with the index:token pairs for the inputs of the model.
            
        out_vocabulary: dictionary
            Vocabulary with the token:index pairs for the outputs of the model.
            
        max_sequence_length: int
            Maximum sequence length accepted by the model, both for the encoder and the decoder.
            
        embedding_dimension: int
            Dimension of the embeddings of the model.
            
        feedforward_dimension: int
            Dimension of the feedforward network inside the self-attention layers of the model.
            
        encoder_layers: int
            Hidden layers of the encoder.
            
        decoder_layers: int
            Hidden layers of the decoder.
            
        attention_heads: int
            Attention heads inside every self-attention layer of the model.
            
        activation: string
            Activation function of the feedforward network inside the self-attention layers of the model. Can
            be either 'relu' or 'gelu'.
            
        dropout: float between 0.0 and 1.0
            Dropout rate to apply to whole model.
        """
        super().__init__()
        self.source_index = source_index
        self.target_index = target_index
        self.source_embeddings = nn.Embedding(len(source_index), embedding_dimension)
        self.target_embeddings = nn.Embedding(len(target_index), embedding_dimension)
        self.positional_embeddings = nn.Embedding(max_sequence_length, embedding_dimension)
        self.transformer = nn.Transformer(d_model = embedding_dimension, 
                                          dim_feedforward = feedforward_dimension,
                                          nhead = attention_heads, 
                                          num_encoder_layers = encoder_layers, 
                                          num_decoder_layers = decoder_layers,
                                          activation = activation,
                                          dropout = dropout)
        self.output_layer = nn.Linear(embedding_dimension, len(target_index))
        self.architecture = dict(model = "Seq2Seq Transformer",
                                 source_index = source_index,
                                 target_index = target_index,
                                 max_sequence_length = max_sequence_length,
                                 embedding_dimension = embedding_dimension,
                                 feedforward_dimension = feedforward_dimension,
                                 encoder_layers = encoder_layers,
                                 decoder_layers = decoder_layers,
                                 attention_heads = attention_heads,
                                 activation = activation,
                                 dropout = dropout)
        self.print_architecture()
        
    def forward(self, X, Y):
        """
        Forward method of the model.
        
        Parameters
        ----------
        X: LongTensor of shape (batch_size, input_length)
            Tensor of integers containing the inputs for the model.
            
        Y: LongTensor of shape (batch_size, output_length)
            Tensor of integers containing the output produced so far.
            
        Returns
        -------
        output: FloatTensor of shape (batch_size, output_length, len(out_vocabulary))
            Tensor of floats containing the inputs for the final Softmax layer (usually integrated in the loss function).
        """
        assert X.shape[1] <= self.architecture["max_sequence_length"]
        assert Y.shape[1] <= self.architecture["max_sequence_length"]
        X = self.source_embeddings(X)
        X_positional = torch.arange(X.shape[1], device = X.device).repeat((X.shape[0], 1))
        X_positional = self.positional_embeddings(X_positional)
        X = (X + X_positional).transpose(0, 1)
        Y = self.target_embeddings(Y)
        Y_positional = torch.arange(Y.shape[1], device = Y.device).repeat((Y.shape[0], 1))
        Y_positional = self.positional_embeddings(Y_positional)
        Y = (Y + Y_positional).transpose(0, 1)
        mask = self.transformer.generate_square_subsequent_mask(Y.shape[0]).to(Y.device)
        transformer_output = self.transformer.forward(src = X,
                                                      tgt = Y, 
                                                      tgt_mask = mask)
        transformer_output = transformer_output.transpose(0, 1)
        return self.output_layer(transformer_output)
    
def add_formatted_columns(datain,
                         source='aligned sentences source',
                         target='aligned sentences target',
                         sent_types='aligned sentences target types'):
    source = []
    target = []
    source_aligned = []
    target_aligned = []
    for i in range(len(datain)):
        d = datain.iloc[i]
        s = np.array(list(d[source])) # aligned source, with ^ symbols
        t = np.array(list(d[target])) # aligned target, with @ symbols
        a = np.array(list(get_fill_in_types(d[sent_types])))
        ss = "".join(s[np.where( (a == ' ') | (a == 'W') | (a == 'w'))[0]].tolist())
        tt = "".join(t[np.where( (a == ' ') | (a == 'W') | (a == 'w'))[0]].tolist())
        source_aligned.append(ss.replace('^','@')) # align with original 
        target_aligned.append(tt)
        source.append(ss.replace('^',''))
        target.append(tt.replace('@',''))

    datain['words source aligned'] = source_aligned
    datain['words target aligned'] = target_aligned
    datain['words source'] = source
    datain['words target'] = target
    return datain

def levenshtein(reference, hypothesis, progress_bar = False):
    #assert len(reference) == len(hypothesis)
    text = zip(reference, hypothesis)
    if progress_bar:
        text = tqdm(text, total = len(reference))
    d = [distance(r, h) for r, h in text]
    output = pd.DataFrame({"reference":reference, "hypothesis":hypothesis})\
    .assign(distance = lambda df: d)\
    .assign(
        cer = lambda df: df.apply(
            lambda r: 100 * r["distance"] / max(len(r["reference"]), 1), 
            axis = 1
        )
    )
    return output

from math import exp

def uniform(j, window_size):
    return 1.0

def triangle(j, window_size):
    m = window_size//2
    return m - 0.5 * abs(m - j)

def bell(j, window_size):
    m = window_size // 2
    s = window_size // 2
    return exp(-((m-j)/s)**2)

def disjoint(
    string,
    model, 
    source_index,
    target_index,
    window_size = 50,
    decoding_method = "greedy_search", 
    document_progress_bar = False, 
    document_batch_progress_bar = 0, 
    use_cuda = True,
    *arcorrect
):
    model.eval()
    windows = [string[i:i+window_size] 
        for i in range(0, len(string), window_size)]
    windows = ["".join([source_index.vocabulary.lookup(c) for c in s])\
        .replace("<UNK>", " ") for s in windows]
    if use_cuda:
        X = source_index.text2tensor(windows, progress_bar = False).cuda()
    else:
        X = source_index.text2tensor(windows, progress_bar = False).cpu()
    if decoding_method == "greedy_search":
        predictions, probs = seq2seq.greedy_search(
            model,
            X, 
            predictions = window_size, 
            progress_bar = document_batch_progress_bar, 
            *arcorrect
        )
    elif decoding_method == "beam_search":
        predictions, probs = seq2seq.beam_search(
            model,
            X, 
            predictions = window_size, 
            progress_bar = document_batch_progress_bar, 
            *arcorrect
        )   
        predictions = predictions[:, 0, :]
    output = target_index.tensor2text(predictions)
    output = [re.sub(r"<START>|<PAD>|<UNK>|<END>.*", "", s) for s in output]
    return "".join(output)

def n_grams(
    string,
    model,
    source_index,
    target_index,
    window_size = 50, 
    decoding_method = "greedy_search", 
    weighting = "uniform",
    document_progress_bar = False, 
    document_batch_progress_bar = 0,      
    main_batch_size = 1024,
    use_cuda=True,
    *arcorrect
):
    model.eval()
    if len(string) <= window_size:
        windows = [string]
    else:
        windows = [string[i:i + window_size] 
            for i in range(len(string) - window_size + 1)]
    windows = ["".join([source_index.vocabulary.lookup(c) for c in s])\
    .replace("<UNK>", " ") for s in windows]
    if use_cuda:
        X = source_index.text2tensor(windows, progress_bar = False).cuda()
    else:
        X = source_index.text2tensor(windows, progress_bar = False).cpu()
    if decoding_method == "greedy_search":
        predictions, probs = seq2seq.greedy_search(
            model,
            X, 
            predictions = window_size, 
            progress_bar = False, 
            *arcorrect
        )
    if decoding_method == "beam_search":
        predictions, probs = seq2seq.beam_search(
            model,
            X, 
            predictions = window_size, 
            progress_bar = document_batch_progress_bar, 
            *arcorrect
        )   
        predictions = predictions[:, 0, :]   
    output = target_index.tensor2text(predictions)
    output = [re.sub(r"<START>|<PAD>|<UNK>|<END>.*", "", s) for s in output]
    if weighting == "uniform":
        weighting = uniform
    elif weighting == "triangle":
        weighting = triangle
    elif weighting == "bell":
        weighting = bell
    votes = [
        {k:0.0 for k in target_index.vocabulary} 
        for c in string
    ]
    for i, s in enumerate(output):
        for j, (counter, char)\
        in enumerate(zip(votes[i:i + window_size], s)):
            counter[char] += weighting(j, window_size)
    output = [max(c.keys(), key = lambda x: c[x]) for c in votes]
    output = "".join(output)
    return votes, output

######################################################
char2i = pickle.load(open(output_folder + "data/char2i_new_pages"+ender+".pkl", "rb"))
i2char = pickle.load(open(output_folder + "data/i2char_new_pages"+ender+".pkl", "rb"))

# what device are we on?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model.to(device)
#predictions.to(device)
use_cuda = True
if device == 'cpu': use_cuda = False
print('device info:', device, use_cuda)

model = Transformer(char2i, 
                    i2char, 
                    max_sequence_length = 110,
                    embedding_dimension = 512,
                    feedforward_dimension = 2048,
                    attention_heads = 8,
                    encoder_layers = 4,
                    decoder_layers = 4)#,
                   #dropout = .5)

    
checkpoint = torch.load(model_save_dir + model_file,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])#,map_location=torch.device('cpu'))
#model.load_state_dict(torch.load(model_save_dir + model_file, map_location=torch.device('cpu')))

print('model loaded')

test_data = pd.read_csv(test_file)

# print('loaded test data')

# test_data = add_formatted_columns(test_data)

# print('format data')

# mask = (pd.isnull(test_data['words source'])) | (pd.isnull(test_data['words target']))

# test_data = test_data[~mask]
# len(mask[mask])

test_data = test_data.rename(columns={'ocr_clean':'ocr_to_input', 
                                      'ocr':'ocr_aligned',
                                      'gt':'gs_aligned'})

test_data = test_data.assign(source = lambda df: df.ocr_aligned.str.replace("@", ""))

print('test data formatting')

ocr_list = test_data["ocr_aligned"].tolist()
gs_list = test_data["gs_aligned"].tolist()

source = []#yy
for s in model.source_index:
    sentence = ""
    for c in s:
        #nu = source_index[c]
        #sentence.append(c)
        #sentence.append(num)
        sentence += c
    source.append(sentence)
    
new_source = []
n = 20 # chunk length  
ntest2 = 100
for x in range(min([48,ntest2])):#only process n paragraphs?
    s = ocr_list[x]
    chunks = [s[i:i+n] for i in range(0, len(s), n)]
    for sentence_chunk in chunks:
        new_source.append(list(sentence_chunk))#append array to new_source array
    new_source.append( list("advice charm touch  "))#new end of sentence with 20 chars
    #because the model only allows 20 characters at a time, each row is 20 chars chunks
    
print('final new source')
    
from pytorch_beam_search import seq2seq
source_index = seq2seq.Index(source)
target_index =  source_index
X_new = source_index.text2tensor(new_source)

model.to(device)

print('model to device')
if not use_cuda:
    predictions, log_probabilities = beam_search(
        model,
        X_new.cpu())
else:
    predictions, log_probabilities = beam_search(
        model,
        X_new.cuda())
    
output = [target_index.tensor2text(p) for p in predictions]

predictions.to(device)

print('predictions to device')

if (window_type == 'n-grams') and (decoding == 'beam') and (weighting == 'uniform'):
    ## beam search
    print("    beam...")
    print("      uniform...")
    corrections = []
    start = time.time()
    for iis,s in enumerate(ocr_list):
        if iis % 100 == 0:
            print('on', iis, 'of', len(ocr_list))
        c = n_grams(
            s,
            model,
            source_index,
            target_index,
            decoding_method = "beam_search", 
            weighting = uniform, 
            document_progress_bar = False,
            window_size = window_size,
            use_cuda=use_cuda
        )[1] 
        corrections.append(c)

    with open(eval_save_dir + 'evalsavecorrs.pickle', 'wb') as f:
        pickle.dump(corrections,f)