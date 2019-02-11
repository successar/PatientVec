import os
import pickle
import re
import shutil
import sys
import json

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import torch
from IPython.core.display import HTML, display
from tqdm import tqdm_notebook

import logging
logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)

np.set_printoptions(suppress=True)

def permute_list(l, p) :
    return [l[i] for i in p]

def calc_max_attn(X, attn) : 
    return np.array([max(attn[i][1:len(X[i])-1]) for i in range(len(attn))])

#########################################################################################################

def plot_entropy(X, attn) :
    unif_H, attn_H = [], []
    for i in range(len(X)) :
        L = len(X[i])
        h = attn[i][1:L-1]
        a = h * np.log(np.clip(h, a_min=1e-8, a_max=None))
        a = -a.sum()
        unif_H.append(np.log(L-2))
        attn_H.append(a)

    plt.scatter(unif_H, attn_H, s=1)

def print_attn(sentence, attention, idx=None, latex=False) :
    l = []
    latex_str = []
    for i, (w, a) in enumerate(zip(sentence, attention)) :
        w = re.sub('&', '&amp;', w)
        w = re.sub('<', '&lt;', w)
        w = re.sub('>', '&gt;', w)
        
        add_string = ''
        if idx is not None and i == idx :
            add_string = "border-style : solid;"
        
        v = "{:.2f}".format((1-a) * -0.5 + 0.5)
        l.append('<span style="background-color:hsl(202,100%,' + str((1-a) * 50 + 50) + '%);' + add_string + '">' + w + ' </span>')
        latex_str.append('{\\setlength{\\fboxsep}{0pt}\\colorbox[Hsb]{202, ' + v + ', 1.0}{\\strut ' + w + '}}')
    
    display(HTML(''.join(l)))
    if latex : 
        return " ".join(latex_str)
    else :
        return ""

def collapse_and_print_word_attn(vocab, doc, word_attn) :
    print("Word Attention " + '#'*10)
    for x, y in zip(doc, word_attn) :
        sent = vocab.map_to_words(x)
        attn = y
        assert len(sent) == len(attn)
        print_attn(sent, attn)

def print_sent_attn(vocab, doc, sent_attn) :
    print("Sent Attention " + '#'*10)
    for x, y in zip(doc, sent_attn) :
        sent = vocab.map_to_words(x)
        attn = [y for _ in range(len(x))]
        assert len(sent) == len(attn)
        print_attn(sent, attn)

#############################################################################################

def pdump(model, values, filename) :
    pickle.dump(values, open(model.dirname + '/' + filename + '_pdump.pkl', 'wb'))

def pload(model, filename) :
    file = model.dirname + '/' + filename + '_pdump.pkl'
    if not os.path.isfile(file) :
        raise FileNotFoundError(file + " doesn't exist")

    return pickle.load(open(file, 'rb'))

def push_graphs_to_main_directory(model, name) :
    dirname = model.dirname
    files = os.listdir(dirname)
    files = [f for f in files if f.endswith('pdf')]
    
    for f in files :
        outdir = f[:-4]
        os.makedirs('graph_outputs/' + outdir, exist_ok=True)
        shutil.copyfile(model.dirname + '/' + f, 'graph_outputs/' + outdir + '/' + outdir + '_' + name + '.pdf')

    files = os.listdir(dirname)
    files = [f for f in files if f.endswith('csv')]
    
    for f in files :
        outdir = f[:-4]
        os.makedirs('graph_outputs/' + outdir, exist_ok=True)
        shutil.copyfile(model.dirname + '/' + f, 'graph_outputs/' + outdir + '/' + outdir + '_' + name + '.csv')

import time
from collections import defaultdict

def push_latest_model(dirname, model_name) :
    exps = defaultdict(list)
    for (path, dirs, files) in os.walk(dirname) :
        if 'evaluate.json' in files :
            exps[os.path.dirname(path)].append((os.path.basename(path), dirs, files))

    for e in exps :
        last_e = max(exps[e], key=lambda s : time.strptime(s[0].replace('_', ' ')))
        evaluate = json.load(open(e + '/' + last_e[0] + '/evaluate.json'))
        evaluate = {'model_name' : model_name, 'time_str' : last_e[0], 'path' : e + '/' + last_e[0], 'results' : evaluate}
        filename = 'latex_evals/' + model_name + '.evaluate.json'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        json.dump(evaluate, open(filename, 'w'))