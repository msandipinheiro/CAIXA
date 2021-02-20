#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Criado em 05/05/2020 às 15:30

@Autor: Marcello Pinheiro
"""

# Bibliotecas utilizadas na manipulação do dataset
import pandas as pd
import numpy as np

# Para tratar as palavras dos textos
import unicodedata
import re

import matplotlib.pyplot as plt
from wordcloud import WordCloud


#%%
# Função que identifica o charset de qualquer dataset.
'''
import chardet
def find_encoding(fname):
    r_file = open(fname, 'rb').read()
    result = chardet.detect(r_file)
    charenc = result['encoding']    
    return charenc

#encoding_termo = find_encoding('sicli_sidratab.csv')
encoding_termo = find_encoding('sicli_sidra_tab_var.csv')
encoding_termo
'''
#%%
# Carregando o dataset em uma estrutura DataFrame do Pandas
# Corpus é um termo utilizado na NLP que exprime uma representação textual

#corpus = pd.read_csv(r'sicli_sidratab.csv', encoding=encoding_termo)
#corpus = pd.read_csv(r'sicli_sidratab.csv', encoding='utf-8')
corpus = pd.read_csv(r'sicli_sidra_tab_var.csv', encoding='utf-8')
#corpus = pd.read_csv(r'sicli_sidra.csv', encoding='utf-8')

# Mostra as cinco primeiras linhas do dataset
termos = pd.DataFrame(data=corpus,copy=True)
#termos.head()

# Carrega do arquivo a lista de stopwords
stopwords = open("stopwords.txt", "r")
stopwords = stopwords.read()
stopwords = stopwords.splitlines()

#%%
def clean_stopwords(text):
    text = text['DESCRICAO'].str.lower()
    text = text.str.split()
    text = text.dropna() # remove nan (not a number)

    #exp = r"\/|\[|\(|\)|\.|-|\]|\+"
    #text = re.sub(exp, '', text)
    #text = re.sub("Þþ", '', text)
    
    text_cleaned = []
    for w in text:
        lista = []
        for l in w:
            if l in stopwords:
                pass
            else:
                lista.append(l)
        text_cleaned.append(lista)

    return text_cleaned
        
#%%
def do_clean(df):
    lista_frases = []
    for t in clean_stopwords(df):
        frase = str()
        for x in t:
            frase += x + ' '
        lista_frases.append(frase)
    return lista_frases

#%%
def normaliza(old):
    new = ''.join(ch for ch in unicodedata.normalize('NFKD', 
                    old) if not unicodedata.combining(ch))
    
    return new

def do_norm(df):        
    ## INCLUIR A COLUNA CODIGO QUANDO USAR O NOVO DATASET CONTENDO O CODIGO
    lista = []
    for t in df:
        lista.append(normaliza(t))
    
    return lista

#%%
# =============================================================================
# Tratamento dos textos usando spaCy
# =============================================================================
import spacy
# pt_core_news_sm é um core do spaCy para português
nlp = spacy.load("pt_core_news_sm")

#%%
def do_lemma(df):
    phrase_list = []
    for t in df.DESCRICAO:
        phrase = []
        doc = nlp(t, disable=['ner','textcat', 'tagger'])
        for w in doc:
            #if not w.is_stop and not w.is_punct and not w.like_num:
            #    phrase.append(w.lemma_)
            phrase.append(w.lemma_)
        phrase_list.append(phrase)

    return df

#%%
# =============================================================================
# Faz a contagem de cada palavra
# =============================================================================
def gen_freq(df):
    #Lista de termos
    #text = df.DESCRICAO.str
    text = df.DESCRICAO
    word_list = []
    frase = []
    lst = str()

    # Loop sobre todos os textos e extrai os termos para dentro de word_list
    for words in text:
        lst = str()
        for word in words.split():
            lst = word
            frase.append(lst)
    word_list.extend(frase)

    #Cria frequência dos termos usando a word_list
    word_freq = pd.Series(word_list).value_counts()

    #Imprime os top 20 termos
    word_freq[:20]
    
    return word_freq 

#%%
# =============================================================================
# Frequencia das palavras antes da "higienização"
# =============================================================================
#gen_freq(termos)

#%%
# =============================================================================
# Gera a nuvem gráfica das palavras mais frequentes
# =============================================================================
def do_nuvem(df):
    word_freq = gen_freq(df)

    #Gerar a nuvem de termos
    wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
#%%
# =============================================================================
# Nuvem gŕafica ANTES da "higienização" do dataset
# =============================================================================
do_nuvem(termos)

#%%
# =============================================================================
# Retira stopwords, normaliza e aplica lemmatização sobre o Corpus
# =============================================================================

termos = do_clean(termos)
termos = do_norm(termos)
termos = pd.DataFrame(termos)
termos = termos.rename(columns={0: 'DESCRICAO'})
termos = do_lemma(termos)

#%%
# =============================================================================
# Nuvem gŕafica DEPOIS da "higienização" do dataset
# =============================================================================
do_nuvem(termos)

#%%
# =============================================================================
# - Cria um dicionário de termos, pré-processa e aplica a mensuração TF/IDF.
# - Os índices criados permitem uma estrutura otimizada para a realização
#   dos cálculos.
# =============================================================================
global dictionary
from gensim import corpora

texts = []
texts = termos.DESCRICAO.str.split()
dictionary = corpora.Dictionary(texts)

#%%
# Remover todos os termos do dicionário que são:
# - menor freq. em 10 frases (ou documento) e
# - maior freq. em 100% do total de frases (ou documentos)

# =============================================================================
# Gera log das palavras pouco frequentes e muito frequentes
# =============================================================================

#def do_log_exlist(dic):
dictionary.filter_extremes(no_below=10, no_above=1)
arq = open('extreme_list.txt', 'w')
qtd = 0
for t in dictionary.token2id:
    #arq.write("'" + t + "'" + ',')
    arq.write(t + "\n")
    qtd +=1
print('Total de palavras removidas: %d' % qtd)
arq.close()

#%%
# Remover todos os termos do dicionário que são:
# - menor freq. em 10 frases (ou documento) e
# - maior freq. em 100% do total de frases (ou documentos)
dictionary.filter_extremes(no_below=10, no_above=1)
exlist = []
qtd = 0
for t in dictionary.token2id:
    exlist.append(t)
    qtd +=1
print('Total de palavras removidas: %d' % qtd)

#%%
def clean_extreme_words(df, exlist):
    text_cleaned = []
    for w in df:
        lista = []
        for l in w:
            if l in exlist:
                pass
            else:
                lista.append(l)
        text_cleaned.append(lista)

    return text_cleaned

#%%
termos_ = clean_extreme_words(termos.DESCRICAO, exlist)

# =============================================================================
# Fim tratamento dos textos
# =============================================================================

#%%
# realiza o pré-processamento dos textos do dataset 
# "quebrando" cada palavra de uma linha em tokens 
from gensim.utils import simple_preprocess

preproc_termos = termos.DESCRICAO.apply(lambda x: simple_preprocess(str(x)))

#%%
''' Treinamento da word2vec sobre o dataset de termos, tendo como 
resultado um acréscimo de palavras/termos de negócio com as do 
modelo Word2Vec pré-treinado '''

# Carrega Word2Vec do Gensim
from gensim.models import Word2Vec
w2v_model = Word2Vec(preproc_termos, size=100, min_count=2, sg=1, workers=8)

#%%
# tamanho do vocabulário
print('Tamanho do vocabulário:', len(w2v_model.wv.vocab))

#%%
# recupera a palavra 'renda' no w2v treinado
print('Dimensão do termo \'renda\' no vetor:', w2v_model.wv.get_vector('renda').shape)

#%%
# procura a palavra que mais se assemelha a 'renda'
words_renda = w2v_model.wv.most_similar('renda')
print(words_renda)

#%%
# procura o termo que mais se assemelha a 'cliente'
words_cliente = w2v_model.wv.most_similar('cliente')
print(words_cliente)

#%%
# gera uma lista de termos similares
words_taxa = w2v_model.wv.most_similar('taxa')
words_cliente = w2v_model.wv.most_similar('cliente')
words_renda = w2v_model.wv.most_similar('renda')

# combina a lista de termos
words = words_taxa + words_cliente + words_renda
print(words)

# extrai somente os termos e não a pontuação de similaridade
words = list(map(lambda x: x[0], words))
print(words)

#%%
'''# Função que utiliza PCA para reduzir o Corpus em duas dimensões,
# no intuito de plotar no plano carteziano X,Y as distâncias entre
# os termos pesquisados no vocabulário. '''

from matplotlib import pyplot
from sklearn.decomposition import PCA

# plots w2v embeddings (incoporação) de uma determinada lista de termos
def plot_w2v(word_list):
    X = w2v_model[word_list]
    
    # reduz para 2 dimensões 
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    
    # cria o scatter plot da projeção
    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(word_list):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        
    pyplot.figure(figsize=(30,61))
    pyplot.show()
    
plot_w2v(words)

#%%
# positive=['renda','cliente','banco']
# negative=['tenha','um','não','sim','nao','dos','se','qual','das','para','da','no']
try:
    result1 = w2v_model.most_similar(positive=['renda','cliente'], 
                                    negative=['banco'], topn=10)
    for r in result1:
        print(r, '\n')
except Exception as inst:
    print('.', inst='')
    
#%%
# positive=['cliente', 'taxa', 'receita', 'renda', 'banco']
# negative=['outro', 'feita','adicionado']
try:
    result2 = w2v_model.most_similar(positive=['cliente', 'taxa'], 
                                    negative=['receita'], topn=10)
    for r in result2:
        print(r, '\n')
except Exception as inst:
    print(inst)

#%%
# positive=['taxa', 'renda','cliente', 'juros']
# negative=['hora','aquela','um']
try:
    result3 = w2v_model.most_similar(positive=['taxa', 'renda','cliente', 'juros'], 
                                    negative=['hora','aquela','um'], topn=10)
    for r in result3:
        print(r, '\n')
except Exception as inst:
    print(inst)

#%%
result4 = w2v_model.get_vector('renda')
print(result4)