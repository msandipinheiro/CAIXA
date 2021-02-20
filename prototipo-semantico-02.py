#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Criado em 05/05/2020 às 15:30

@Autores: Marcello Pinheiro e Thiago Gomes
"""
# Para tratar as palavras dos textos
import unicodedata # Conjunto de caracteres de uniformidade unicode
import re # Regular Expression

import os # para execução de comandos no sistema operacional
import argparse # para passar parâmetros em __main__

# Remove mensagens de libraries com funções que serão descontinuadas
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# Gensim Word2Vec
# https://radimrehurek.com/gensim/index.html
from gensim.models import Word2Vec

# Gera o log no console durante o treinamento do modelo
# Word2Vec com o dicinário de termos do SIDRA e/ou SICLI
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# pt_core_news_sm é um core NLP do spaCy para português
# https://spacy.io/usage/models
import spacy
nlp = spacy.load("pt_core_news_sm")

# Library Pandas para manipulação de arquivos em diversos formatos
# https://pandas.pydata.org/
import pandas as pd

# 8.3. collections — High-performance container datatypes
# https://docs.python.org/2/library/collections.html
from collections import defaultdict

# Variáveis globais que serão utilizadas ao longo do programa
global texts
global stoplist
#global nlp
global dictionary
global w2v_model

#%%
# Identifica o charset do dataset de input
# A depender do tamanho do arquivo pode ficar bem lenta a leitura
# de identificação do charset

import chardet
def find_encoding(fname):
    r_file = open(fname, 'rb').read()
    result = chardet.detect(r_file)
    charenc = result['encoding']    
    return charenc

#%%
# Função para ler o dataset
#def dataset_read(dataset, encod):
def dataset_read(dataset):
    #df = pd.read_csv(r''+dataset, encoding=encod, sep=';')
    #df = pd.read_csv(r''+dataset, encoding='utf-8', sep=';')
    df = pd.read_csv(dataset, delimiter=';', encoding='UTF-8', header=None)
    return df

#%%
# Carrega o arquivo com o conjunto de palavras indesejadas
def load_stoplist():
    stoplist = open("stopwords.txt", "r")
    stoplist = stoplist.read()
    stoplist = set(stoplist.splitlines())
    return stoplist

# Remove palavras comuns contidas na stoplist
def do_clean_stoplist(df):
    df[8] = df[8].dropna()
    list_words_clean = []
    for words in df[8]:
        list_words = []
        words = words.lower()
        words = words.split()
        for word in words:
            word = re.sub(r"\W", '', word)
            word = re.sub(r"\d", '', word)
            if word not in stoplist and len(word) > 2:
                list_words.append(word)
        list_words_clean.append(list_words)
    return list_words_clean

#%%
# =============================================================================
# Aplica o conjunto unicode NFKD Compatibility Decomposition
# https://unicode.org/reports/tr15/
# =============================================================================

def norma(old):
    new = ''.join(ch for ch in unicodedata.normalize('NFKD', 
                    str(old)) if not unicodedata.combining(ch))
    
    return new

def do_norma(txt):        
    ## INCLUIR A COLUNA CODIGO QUANDO USAR O NOVO DATASET CONTENDO O CODIGO
    sentence_list = []
    for sentence in txt:
        phrase = []    
        for word in sentence:
            phrase.append(norma(word))
        sentence_list.append(phrase)
    return sentence_list 

#%%
# =============================================================================
# Tratamento dos textos usando spaCy
# Aplicando a técnica de lemmatização
# https://en.wikipedia.org/wiki/Lemmatisation
# =============================================================================
def do_lemma(text):
    sentence_list = []
    #doc = nlp(str(text), disable=['ner','textcat', 'tagger'])
    doc = nlp(str(text))
    # for t in text:
    #     sentence = []
    #     #doc = nlp(str(t), disable=['ner','textcat', 'tagger'])
    #     for w in doc:
    #         #if not w.is_stop and not w.is_punct and not w.like_num:
    #             #sentence.append(w.lemma_)
    #         sentence.append(w.lemma_)
    #     sentence_list.append(sentence)
    #for sent in doc.
    for sent in doc.
    
    return sentence_list

#%%
doc = nlp(str(text), disable=['ner','textcat', 'tagger'])


#%%
# Remove palavras que aparecem 'n' vezes

def do_clean_low_freq(txt, n):
    frequency = defaultdict(int)
    for text in txt:
        for token in text:
            frequency[token] += 1
    
    txt = [
        [token for token in text if frequency[token] > n]
        for text in txt
    ]
    return txt

#%%
# Carrega Word2Vec do Gensim
def load_w2v(txt, procs):
    """ procs = multiprocessing.cpu_count() """
    w2v_model = Word2Vec(txt, size=100, min_count=2, sg=1, workers=procs)
    #w2v_model = Word2Vec(txt, size=100, min_count=2, sg=1, workers=8)
    return w2v_model

#%%
def find_most_similarity(w2v_model_, lista_pos, lista_neg, top_n):
    try:
        result = w2v_model_.most_similar(positive=lista_pos, 
                                        negative=lista_neg, topn=top_n)
        for r in result:
            print(r, '\n')
    except Exception as inst:
        print('.', inst='')
    return result

#%%
texts = dataset_read('df_amostra_sicli.csv')

stoplist = load_stoplist()

texts = do_clean_stoplist(texts)

texts = do_norma(texts)

#%%
texts = do_lemma(texts)

do_clean_low_freq(texts, 2)

w2v_model = load_w2v(texts, 8)

#%%
# Grava em disco/ssd o modelo que foi treinado
w2v_model.save('w2v_model.pkl')

#%%
# Recupera para uso o modelo que foi treinado
reload_w2v = Word2Vec.load('w2v_model.pkl')

#semantic_words = find_most_similarity(reload_w2v, ['renda', 'inpc', 'custo'], ['geladeira', 'freezer', 'microondas'], 10)

#%%
dataset = dataset_read('Tabelas-sidra-short.csv')
for dts in dataset.metadados:
    line = 1
    word_pos = do_clean_stoplist(dts)
    word_pos = do_norma(word_pos)
    #word_pos = do_lemma(word_pos)

    lista_neg = ['geladeira', 'freezer', 'microondas']
    word_neg = do_clean_stoplist(lista_neg)
    word_neg = do_norma(word_neg)
    #word_neg = do_lemma(word_neg)

    result = find_most_similarity(reload_w2v, word_pos, word_neg, 10)
    print('Id tabela SICLI: ', dts.id)
    print('Linha do dataset: %', line, '\nTermos similares: \n', result)
