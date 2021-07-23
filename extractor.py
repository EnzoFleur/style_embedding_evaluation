from __future__ import division
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import map_tag
from collections import Counter
import nltk
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import spacy
from spacy.lang.en import English

import re
import pickle
import math
import argparse
 
import pandas as pd
import time
from tqdm import tqdm

np.random.seed(13)

nltk.download('cmudict')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

cmuDictionary = nltk.corpus.cmudict.dict()
function_word = set(stopwords.words('english'))
with open('dale-chall.pkl', 'rb') as dalechall:
    familiarWords = pickle.load(dalechall)

def clean_str(string):
    string= re.sub(r"[^A-Za-z0-9(),.:;$!?%#\{\}\[\]\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
def read(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f_in:
        content=re.sub('\r\n', '', f_in.read())
        content=clean_str(content)
    return(content)

def tokenize(text):  # tokenize the text
    tokens = nltk.word_tokenize(text)
    return tokens

# --- Lexical Features --- #

# Lexical feature (word level)
# Average word length of a document
def average_word_length(words):
    # word_list = text_doc.split(" ")
    if len(words)==0:
        return 0
    average = sum(len(word) for word in words)/(len(words))
    return average

# Total number of short words (length <4) in a document
def total_short_words(words):
    # word_list = text_doc.split(" ")
    count_short_word = 0
    if len(words)==0:
        return 0
    for word in words:
        if len(word) < 4:
            count_short_word += 1
    return count_short_word/(len(words))

# Lexical feature (character level)
# Average number of digit in document
def total_digit(text_doc):
    return sum(c.isdigit() for c in text_doc)/(len(text_doc))

# Average number of uppercase letters in document
def total_uppercase(text_doc):
    return sum(1 for c in text_doc if c.isupper())/(len(text_doc))

# Letter frequency in document
def count_letter_freq(text_doc):
    text_doc = ''.join([i.lower() for i in text_doc if i.isalpha()])
    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's'
              , 't', 'u', 'v', 'w', 'x', 'y', 'z']
    count = {}
    text_length=len(text_doc)
    for s in text_doc:
      if s in count.keys():
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in letter:
        if d in count.keys():
            count_list[d] = count[d]
        else:
            count_list[d] = 0
    return({ll:count[ll]/text_length if ll in count.keys() else 0 for ll in letter})

# Digit frequency in document
def count_digit_freq(text_doc): 
    text_doc = ''.join([i for i in text_doc if i.isdigit()])
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    count = {}
    text_length = len(text_doc)
    for s in text_doc:
      if s in count.keys():
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in digits:
        if d in count.keys():
            count_list[d] = count[d]
        else:
            count_list[d] = 0

    return({dd:count[dd]/text_length if dd in count.keys() else 0 for dd in digits})

# Average sentence length in a document
def average_sentence_length(sent_list):
    # sent_list = sent_tokenize(text_doc, language='english')
    average = sum(len(tokenize(sent)) for sent in sent_list)/(len(sent_list))
    return average


# Lexical Feature (vocabulary richness)
def hapax_legomena_ratio(words):  # # per document only a float value
    # word_list = text.split(" ")
    fdist = nltk.FreqDist(word for word in words)
    fdist_hapax = nltk.FreqDist.hapaxes(fdist)
    return float(len(fdist_hapax)/(len(words)))


def dislegomena_ratio(words):  # per document only a float value
    # word_list = text.split(" ")
    vocabulary_size = len(set(words))
    freqs = Counter(nltk.probability.FreqDist(words).values())
    VN = lambda i:freqs[i]
    return float(VN(2)*1./(vocabulary_size))

def CountFunctionalWords(words):

    # words = RemoveSpecialCHs(text)
    count = 0

    for i in words:
        if i in function_word:
            count += 1

    return count / len(words)

def freq_function_word(words):  # per document (vector with length 174)
    # words = text.split(" ")
    count = {}
    n_words=len(words)
    for s in words:
      if s in count.keys():
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in function_word:
        if d in count.keys():
            count_list[d] = count[d]/n_words
        else:
            count_list[d] = 0
    return {ww:(count_list[ww]) for ww in function_word}

def punctuation_freq(text):
    punct = ['\'', ':', ',', '_', '!', '?', ';', ".", '\"', '(', ')', '-']
    count = {}
    text_length=len(text)
    for s in text:
      if s in count.keys():
        count[s] += 1
      else:
        count[s] = 1
    count_list = {}
    for d in punct:
        if d in count.keys():
            count_list[d] = count[d]
        else:
            count_list[d] = 0

    return({pp:count[pp]/text_length if pp in count.keys() else 0 for pp in punct})

def syllable_count_Manual(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def syllable_count(word):
    try:
        syl = [len(list(y for y in x if y[-1].isdigit())) for x in cmuDictionary[word.lower()]][0]
    except:
        syl = syllable_count_Manual(word)
    return syl

def Avg_Syllable_per_Word(words):
    syllabls = [syllable_count(word) for word in words]
    p = (" ".join(words))
    return sum(syllabls) / max(1, len(words))

def RemoveSpecialCHs(text):
    text = word_tokenize(text)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

    words = [word for word in text if word not in st]
    return words

def AvgWordFrequencyClass(words):
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    maximum = float(max(list(freqs.values())))
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])

# -------------------------------------------------------------------------
# K  10,000 * (M - N) / N**2
# , where M  Sigma i**2 * Vi.
def YulesCharacteristicK(words):
    N = len(words)
    freqs = Counter()
    freqs.update(words)
    vi = Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    K = 10000 * (M - N) / math.pow(N, 2)
    return K


# -------------------------------------------------------------------------


# -1*sigma(pi*lnpi)
# Shannon and sympsons index are basically diversity indices for any community
def ShannonEntropy(words):
    lenght = len(words)
    freqs = Counter()
    freqs.update(words)
    arr = np.array(list(freqs.values()))
    distribution = 1. * arr
    distribution /= max(1, lenght)
    import scipy as sc
    H = sc.stats.entropy(distribution, base=2)
    return H


# ------------------------------------------------------------------
# 1 - (sigma(n(n - 1))/N(N-1)
# N is total number of words
# n is the number of each type of word
def SimpsonsIndex(words):
    freqs = Counter()
    freqs.update(words)
    N = len(words)
    if N<=1:
        return 0
    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    D = 1 - (n / (N * (N - 1)))
    return D


# ------------------------------------------------------------------

def FleschReadingEase(words, NoOfsentences):
    l = float(len(words))
    scount = 0
    for word in words:
        scount += syllable_count(word)

    I = 206.835 - 1.015 * (l / float(NoOfsentences)) - 84.6 * (scount / float(l))
    return I


# -------------------------------------------------------------------
def FleschCincadeGradeLevel(words, NoOfSentences):
    scount = 0
    for word in words:
        scount += syllable_count(word)

    l = len(words)
    F = 0.39 * (l / NoOfSentences) + 11.8 * (scount / float(l)) - 15.59
    return F


# -----------------------------------------------------------------
def dale_chall_readability_formula(words, NoOfSectences):
    difficult = 0
    adjusted = 0
    NoOfWords = len(words)

    for word in words:
        if word not in familiarWords:
            difficult += 1
    percent = (difficult / NoOfWords) * 100
    if (percent > 5):
        adjusted = 3.6365
    D = 0.1579 * (percent) + 0.0496 * (NoOfWords / NoOfSectences) + adjusted
    return D


# ------------------------------------------------------------------
def GunningFoxIndex(words, NoOfSentences):
    NoOFWords = float(len(words))
    complexWords = 0
    for word in words:
        if (syllable_count(word) > 2):
            complexWords += 1

    G = 0.4 * ((NoOFWords / NoOfSentences) + 100 * (complexWords / NoOFWords))
    return G


def pos_freq(text, NoOfSentences):      
    doc = nlp(text)

    features_dict={
        **{k:v/NoOfSentences for k, v in Counter([token.tag_ for token in doc]).items()},
        **{k:v/NoOfSentences for k,v in Counter([entity.label_ for entity in doc.ents]).items()}
    }

    return(features_dict)

def create_feature(text, words, sent_text):

    stylometry = {}
    text = ''.join(sent_text)
    NoOfSentences = len(sent_text)

    stylometry['avg_w_len']=average_word_length(words)
    stylometry['tot_short_w']=total_short_words(words)
    stylometry['tot_digit']=total_digit(text)
    stylometry['tot_upper']=total_uppercase(text)
    stylometry={**stylometry, **count_letter_freq(text)}
    stylometry={**stylometry, **count_digit_freq(text)}
    stylometry['avg_s_len']=average_sentence_length(sent_text)
    stylometry['hapax']=hapax_legomena_ratio(words)
    stylometry['dis']=dislegomena_ratio(words)
    stylometry['func_w_freq']=CountFunctionalWords(words)
    stylometry={**stylometry, **freq_function_word(words)}
    stylometry={**stylometry,**punctuation_freq(text)}
    stylometry["syllable_count"]=Avg_Syllable_per_Word(words)
    stylometry["avg_w_freqc"]=AvgWordFrequencyClass(words)
    stylometry['yules_K']=YulesCharacteristicK(words)
    stylometry['shannon_entr']=ShannonEntropy(words)
    stylometry['simposons_ind']=SimpsonsIndex(words)
    stylometry['flesh_ease']=FleschReadingEase(words, NoOfSentences)
    stylometry['flesh_cincade']=FleschCincadeGradeLevel(words, NoOfSentences)
    stylometry['dale_call']=dale_chall_readability_formula(words, NoOfSentences)
    stylometry['gunnin_fox']=GunningFoxIndex(words, NoOfSentences)
    stylometry={**stylometry, **pos_freq(text, NoOfSentences)}   

    return stylometry

def create_stylometrics(authorship, max_sentence=200):

    stylo_dict = {}

    for (ff, id, author) in tqdm(zip(authorship['file'], authorship['id'], authorship['author']), total=len(authorship)):
        
        text = read(ff)

        sent_text = sent_tokenize(text)[:max_sentence]
        
        text = ' '.join(sent_text)
        words=RemoveSpecialCHs(text)
        
        stylo_dict[(author, id)]=create_feature(text, words, sent_text)

    stylo_df=pd.DataFrame(stylo_dict).transpose().rename_axis(['author', 'id']).reset_index().fillna(0)

    return stylo_df

def build_authorship(dir):
    authorship=[]
    authors = [a for a in os.listdir(dir) if os.path.isdir(os.path.join(dir,a))]
    for author in authors:
        books = [(author,b, os.path.join(dir, author, b)) for b in os.listdir(os.path.join(dir, author))]
        authorship.extend(books)

    authorship_df = pd.DataFrame(authorship, columns=['author', 'id', 'file'])
    return authorship_df