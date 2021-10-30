import numpy as np
from collections import Counter
import nltk
import os
from nltk.corpus import stopwords
import spacy

import re
import pickle
import math
 
import pandas as pd
from tqdm import tqdm
from syllabation import getSyllabation

from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop

from nltk import RegexpTokenizer

import nltk.tokenize.punkt as pkt

word_tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')

postag_keys=['ADJ|',
    'ADJ|Gender=Fem|NumType=Ord|Number=Plur',
    'ADJ|Gender=Fem|NumType=Ord|Number=Sing',
    'ADJ|Gender=Fem|Number=Plur',
    'ADJ|Gender=Fem|Number=Sing',
    'ADJ|Gender=Masc',
    'ADJ|Gender=Masc|NumType=Ord|Number=Plur',
    'ADJ|Gender=Masc|NumType=Ord|Number=Sing',
    'ADJ|Gender=Masc|Number=Plur',
    'ADJ|Gender=Masc|Number=Sing',
    'ADJ|NumType=Ord',
    'ADJ|NumType=Ord|Number=Sing',
    'ADJ|Number=Plur',
    'ADJ|Number=Sing',
    'ADP|',
    'ADP|Definite=Def|Gender=Masc|Number=Sing|PronType=Art',
    'ADP|Definite=Def|Number=Plur|PronType=Art',
    'ADV|',
    'ADV|Gender=Fem',
    'ADV|Polarity=Neg',
    'ADV|PronType=Int',
    'AUX|Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part',
    'AUX|Mood=Cnd|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Cnd|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Plur|Person=1|Tense=Fut|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Plur|Person=1|Tense=Imp|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Plur|Person=2|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Plur|Person=3|Tense=Fut|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Plur|Person=3|Tense=Imp|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Sing|Person=1|Tense=Imp|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Sing|Person=2|Tense=Imp|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Sing|Person=3|Tense=Fut|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Sing|Person=3|Tense=Imp|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin',
    'AUX|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Sub|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Sub|Number=Plur|Person=2|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Sub|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Sub|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin',
    'AUX|Mood=Sub|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin',
    'AUX|Tense=Past|VerbForm=Part',
    'AUX|Tense=Pres|VerbForm=Part',
    'AUX|VerbForm=Inf',
    'CCONJ|',
    'DET|',
    'DET|Definite=Def|Gender=Fem|Number=Sing|PronType=Art',
    'DET|Definite=Def|Gender=Masc|Number=Sing|PronType=Art',
    'DET|Definite=Def|Number=Plur|PronType=Art',
    'DET|Definite=Def|Number=Sing|PronType=Art',
    'DET|Definite=Ind|Gender=Fem|Number=Plur|PronType=Art',
    'DET|Definite=Ind|Gender=Fem|Number=Sing|PronType=Art',
    'DET|Definite=Ind|Gender=Masc|Number=Plur|PronType=Art',
    'DET|Definite=Ind|Gender=Masc|Number=Sing|PronType=Art',
    'DET|Definite=Ind|Number=Plur|PronType=Art',
    'DET|Definite=Ind|Number=Sing|PronType=Art',
    'DET|Gender=Fem|Number=Plur',
    'DET|Gender=Fem|Number=Plur|PronType=Int',
    'DET|Gender=Fem|Number=Sing',
    'DET|Gender=Fem|Number=Sing|Poss=Yes',
    'DET|Gender=Fem|Number=Sing|PronType=Dem',
    'DET|Gender=Fem|Number=Sing|PronType=Int',
    'DET|Gender=Masc|Number=Plur',
    'DET|Gender=Masc|Number=Sing',
    'DET|Gender=Masc|Number=Sing|PronType=Dem',
    'DET|Gender=Masc|Number=Sing|PronType=Int',
    'DET|Number=Plur',
    'DET|Number=Plur|Poss=Yes',
    'DET|Number=Plur|PronType=Dem',
    'DET|Number=Sing',
    'DET|Number=Sing|Poss=Yes',
    'INTJ|',
    'NOUN|',
    'NOUN|Gender=Fem',
    'NOUN|Gender=Fem|Number=Plur',
    'NOUN|Gender=Fem|Number=Sing',
    'NOUN|Gender=Masc',
    'NOUN|Gender=Masc|NumType=Card|Number=Plur',
    'NOUN|Gender=Masc|NumType=Card|Number=Sing',
    'NOUN|Gender=Masc|Number=Plur',
    'NOUN|Gender=Masc|Number=Sing',
    'NOUN|NumType=Card',
    'NOUN|Number=Plur',
    'NOUN|Number=Sing',
    'NUM|',
    'NUM|Gender=Masc|NumType=Card',
    'NUM|NumType=Card',
    'PART|',
    'PRON|',
    'PRON|Gender=Fem',
    'PRON|Gender=Fem|Number=Plur',
    'PRON|Gender=Fem|Number=Plur|Person=3',
    'PRON|Gender=Fem|Number=Plur|Person=3|PronType=Prs',
    'PRON|Gender=Fem|Number=Plur|PronType=Dem',
    'PRON|Gender=Fem|Number=Plur|PronType=Rel',
    'PRON|Gender=Fem|Number=Sing',
    'PRON|Gender=Fem|Number=Sing|Person=3',
    'PRON|Gender=Fem|Number=Sing|Person=3|PronType=Prs',
    'PRON|Gender=Fem|Number=Sing|PronType=Dem',
    'PRON|Gender=Fem|Number=Sing|PronType=Rel',
    'PRON|Gender=Masc',
    'PRON|Gender=Masc|Number=Plur',
    'PRON|Gender=Masc|Number=Plur|Person=3',
    'PRON|Gender=Masc|Number=Plur|Person=3|PronType=Prs',
    'PRON|Gender=Masc|Number=Plur|PronType=Dem',
    'PRON|Gender=Masc|Number=Plur|PronType=Rel',
    'PRON|Gender=Masc|Number=Sing',
    'PRON|Gender=Masc|Number=Sing|Person=3',
    'PRON|Gender=Masc|Number=Sing|Person=3|PronType=Dem',
    'PRON|Gender=Masc|Number=Sing|Person=3|PronType=Prs',
    'PRON|Gender=Masc|Number=Sing|PronType=Dem',
    'PRON|Gender=Masc|Number=Sing|PronType=Rel',
    'PRON|NumType=Card',
    'PRON|Number=Plur',
    'PRON|Number=Plur|Person=1',
    'PRON|Number=Plur|Person=1|PronType=Prs',
    'PRON|Number=Plur|Person=1|Reflex=Yes',
    'PRON|Number=Plur|Person=2',
    'PRON|Number=Plur|Person=2|PronType=Prs',
    'PRON|Number=Plur|Person=2|Reflex=Yes',
    'PRON|Number=Plur|Person=3',
    'PRON|Number=Sing',
    'PRON|Number=Sing|Person=1',
    'PRON|Number=Sing|Person=1|PronType=Prs',
    'PRON|Number=Sing|Person=1|Reflex=Yes',
    'PRON|Number=Sing|Person=2|PronType=Prs',
    'PRON|Number=Sing|Person=3',
    'PRON|Number=Sing|PronType=Dem',
    'PRON|Person=3',
    'PRON|Person=3|Reflex=Yes',
    'PRON|PronType=Int',
    'PRON|PronType=Rel',
    'PROPN|',
    'PROPN|Gender=Fem|Number=Plur',
    'PROPN|Gender=Fem|Number=Sing',
    'PROPN|Gender=Masc',
    'PROPN|Gender=Masc|Number=Plur',
    'PROPN|Gender=Masc|Number=Sing',
    'PROPN|Number=Plur',
    'PROPN|Number=Sing',
    'PUNCT|',
    'SCONJ|',
    'SPACE|',
    'SYM|',
    'VERB|Gender=Fem|Number=Plur|Tense=Past|VerbForm=Part',
    'VERB|Gender=Fem|Number=Plur|Tense=Past|VerbForm=Part|Voice=Pass',
    'VERB|Gender=Fem|Number=Sing|Tense=Past|VerbForm=Part',
    'VERB|Gender=Fem|Number=Sing|Tense=Past|VerbForm=Part|Voice=Pass',
    'VERB|Gender=Masc|Number=Plur|Tense=Past|VerbForm=Part',
    'VERB|Gender=Masc|Number=Plur|Tense=Past|VerbForm=Part|Voice=Pass',
    'VERB|Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part',
    'VERB|Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part|Voice=Pass',
    'VERB|Gender=Masc|Tense=Past|VerbForm=Part',
    'VERB|Gender=Masc|Tense=Past|VerbForm=Part|Voice=Pass',
    'VERB|Mood=Cnd|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Cnd|Number=Plur|Person=2|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Cnd|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Cnd|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Imp|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Imp|Number=Plur|Person=2|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Imp|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Plur|Person=1|Tense=Fut|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Plur|Person=1|Tense=Imp|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Plur|Person=2|Tense=Fut|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Plur|Person=2|Tense=Imp|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Plur|Person=2|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Plur|Person=3|Tense=Fut|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Plur|Person=3|Tense=Imp|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Sing|Person=1|Tense=Fut|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Sing|Person=1|Tense=Imp|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Sing|Person=3|Tense=Fut|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Sing|Person=3|Tense=Imp|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin',
    'VERB|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Ind|Person=3|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Ind|Person=3|VerbForm=Fin',
    'VERB|Mood=Ind|VerbForm=Fin',
    'VERB|Mood=Sub|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Sub|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin',
    'VERB|Mood=Sub|Number=Sing|Person=3|Tense=Past|VerbForm=Fin',
    'VERB|Mood=Sub|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin',
    'VERB|Number=Plur|Tense=Past|VerbForm=Part',
    'VERB|Number=Plur|Tense=Past|VerbForm=Part|Voice=Pass',
    'VERB|Number=Sing|Tense=Past|VerbForm=Part',
    'VERB|Number=Sing|Tense=Past|VerbForm=Part|Voice=Pass',
    'VERB|Tense=Past|VerbForm=Part',
    'VERB|Tense=Past|VerbForm=Part|Voice=Pass',
    'VERB|Tense=Pres|VerbForm=Part',
    'VERB|VerbForm=Inf',
    'X|']

class CustomLanguageVars(pkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

sent_tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')
sent_tokenizer_SP = pkt.PunktSentenceTokenizer(lang_vars=CustomLanguageVars())

nlp = spacy.load("fr_core_news_lg")

# Stanza probablement plus performant pour la NER mais plus lent 
# import stanza
# stanza.download('fr')
# nlp = stanza.Pipeline('fr', processors='tokenize,ner')
# doc = nlp(text.replace('\n', ' '))

# doc = nlp("Michel et l'Antilope vont dans l'Yonne le 3 août 2005.")

# print(doc)
# print(doc.entities)

function_word = set(fr_stop)
# with open('dale-chall.pkl', 'rb') as dalechall:
#     familiarWords = pickle.load(dalechall)

def clean_str(string):
    string= re.sub(r"[^œA-zÀ-ÿ0-9!\"£€#$%&’'()*+,-./:;<=>?@[\]^_`{|}~\n]", " ", string)
    string = re.sub("œ", "oe", string)
    string = re.sub(r"\n{2,}", "\n", string)
    return string.strip()
    
def read(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f_in:
        # content=re.sub('\r\n', '', f_in.read())
        content=clean_str(f_in.read())
    return(content)

def word_tokenize(text):  # tokenize the text
    tokens = word_tokenizer.tokenize(text)
    return tokens

def sent_tokenize(text, space=False):
    if space:
        return sent_tokenizer_SP.tokenize(text)
    else:
        return sent_tokenizer.tokenize(text) 

####################################
book_path = "dataset\\Barbusse\\book_1.txt"
content = read(book_path)
sent_content = ''.join(sent_tokenize(content)[:500])
####################################

# --- Lexical Features --- #

# Lexical feature (word level)
# Average word length of a document
def average_word_length(words):
    if len(words)==0:
        return 0
    average = sum(len(word) for word in words)/(len(words))
    return average

# Total number of short words (length <4) in a document
def total_short_words(words):
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
    average = sum(len(word_tokenize(sent)) for sent in sent_list)/(len(sent_list))
    return average


# Lexical Feature (vocabulary richness)
def hapax_legomena_ratio(words):  # # per document only a float value
    fdist = nltk.FreqDist(word for word in words)
    fdist_hapax = nltk.FreqDist.hapaxes(fdist)
    return float(len(fdist_hapax)/(len(words)))


def dislegomena_ratio(words):  # per document only a float value
    vocabulary_size = len(set(words))
    freqs = Counter(nltk.probability.FreqDist(word for word in words).values())
    VN = lambda i:freqs[i]
    return float(VN(2)*1./(vocabulary_size))

def CountFunctionalWords(words):

    count = 0

    for i in words:
        if i in function_word:
            count += 1

    return count / len(words)

def freq_function_word(words):  # per document (vector with length 174)
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
    punct = ["!", "\"", "£", "€", "#", "$", "%", "&", "’", "'", "(", ")", "*", "+", ",", "-", ".",
         "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "\n"]
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

def syllable_count(word):
    return getSyllabation(word, count=True)

def Avg_Syllable_per_Word(words):
    syllabls = [syllable_count(word) for word in words]
    return sum(syllabls) / max(1, len(words))

def RemoveSpecialCHs(text):
    words = word_tokenize(re.sub(r"[^œA-zÀ-ÿ]", " ", text))
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
    doc = nlp(text.replace("\n", " "))

    pos_dict={k:v/NoOfSentences for k, v in Counter([f"{token.tag_}|" if token.tag_ == "SPACE" else f"{token.tag_}|{token.morph}" for token in doc]).items()}
    pos_dict={**pos_dict, **{k:0 for k in postag_keys if k not in pos_dict.keys()}}
    
    ner_dict={k:v/NoOfSentences for k,v in Counter([entity.label_ for entity in doc.ents]).items()}
    ner_dict={**ner_dict, **{k:0 for k in ["MISC","PER","LOC","ORG"] if k not in ner_dict.keys()}}
    
    features_dict={**pos_dict, **ner_dict}

    return(features_dict)

def create_feature(text, words, sent_text):

    low_words = [word.lower() for word in words]
    stylometry = {}
    NoOfSentences = len(sent_text)

    stylometry['avg_w_len']=average_word_length(words)
    stylometry['tot_short_w']=total_short_words(words)
    stylometry['tot_digit']=total_digit(text)
    stylometry['tot_upper']=total_uppercase(text)
    stylometry={**stylometry, **count_letter_freq(text)}
    stylometry={**stylometry, **count_digit_freq(text)}
    stylometry['avg_s_len']=average_sentence_length(sent_text)
    stylometry['hapax']=hapax_legomena_ratio(low_words)
    stylometry['dis']=dislegomena_ratio(low_words)
    stylometry['func_w_freq']=CountFunctionalWords(low_words)
    stylometry={**stylometry, **freq_function_word(low_words)}
    stylometry={**stylometry,**punctuation_freq(text)}
    stylometry["syllable_count"]=Avg_Syllable_per_Word(words)
    stylometry["avg_w_freqc"]=AvgWordFrequencyClass(words)
    stylometry['yules_K']=YulesCharacteristicK(words)
    stylometry['shannon_entr']=ShannonEntropy(words)
    stylometry['simposons_ind']=SimpsonsIndex(words)
    stylometry['flesh_ease']=FleschReadingEase(words, NoOfSentences)
    stylometry['flesh_cincade']=FleschCincadeGradeLevel(words, NoOfSentences)
    # stylometry['dale_call']=dale_chall_readability_formula(words, NoOfSentences)
    stylometry['gunnin_fox']=GunningFoxIndex(words, NoOfSentences)
    stylometry={**stylometry, **pos_freq(text, NoOfSentences)}   

    return stylometry

def create_stylometrics(authorship, max_sentence=300):

    stylo_dict = {}

    for (ff, id, author) in tqdm(zip(authorship['file'], authorship['id'], authorship['author']), total=len(authorship)):
        
        text = read(ff)

        sent_text = sent_tokenize(text, space=True)[:max_sentence]
        
        text = ''.join(sent_text)
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