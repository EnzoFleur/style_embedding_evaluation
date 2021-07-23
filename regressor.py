import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import tqdm
import time

np.seterr(divide='ignore', invalid='ignore')

letters={'a': 'Letters',
    'b': 'Letters',
    'c': 'Letters',
    'd': 'Letters',
    'e': 'Letters',
    'f': 'Letters',
    'g': 'Letters',
    'h': 'Letters',
    'i': 'Letters',
    'j': 'Letters',
    'k': 'Letters',
    'l': 'Letters',
    'm': 'Letters',
    'n': 'Letters',
    'o': 'Letters',
    'p': 'Letters',
    'q': 'Letters',
    'r': 'Letters',
    's': 'Letters',
    't': 'Letters',
    'u': 'Letters',
    'v': 'Letters',
    'w': 'Letters',
    'x': 'Letters',
    'y': 'Letters',
    'z': 'Letters'}
numbers={'tot_digit': 'Numbers',
    '0': 'Numbers',
    '1': 'Numbers',
    '2': 'Numbers',
    '3': 'Numbers',
    '4': 'Numbers',
    '5': 'Numbers',
    '6': 'Numbers',
    '7': 'Numbers',
    '8': 'Numbers',
    '9': 'Numbers'}
ner={'CARDINAL': 'NER',
    'DATE': 'NER',
    'EVENT': 'NER',
    'FAC': 'NER',
    'GPE': 'NER',
    'LANGUAGE': 'NER',
    'LAW': 'NER',
    'LOC': 'NER',
    'MONEY': 'NER',
    'NORP': 'NER',
    'ORDINAL': 'NER',
    'ORG': 'NER',
    'PERCENT': 'NER',
    'PERSON': 'NER',
    'PRODUCT': 'NER',
    'QUANTITY': 'NER',
    'TIME': 'NER',
    'WORK_OF_ART': 'NER'}
tag={'-LRB-': 'TAG',
    '-RRB-': 'TAG',
    'ADD': 'TAG',
    'AFX': 'TAG',
    'CC': 'TAG',
    'CD': 'TAG',
    'DT': 'TAG',
    'EX': 'TAG',
    'FW': 'TAG',
    'HYPH': 'TAG',
    'IN': 'TAG',
    'JJ': 'TAG',
    'JJR': 'TAG',
    'JJS': 'TAG',
    'LS': 'TAG',
    'MD': 'TAG',
    'NFP': 'TAG',
    'NN': 'TAG',
    'NNP': 'TAG',
    'NNPS': 'TAG',
    'NNS': 'TAG',
    'PDT': 'TAG',
    'POS': 'TAG',
    'PRP': 'TAG',
    'PRP$': 'TAG',
    'RB': 'TAG',
    'RBR': 'TAG',
    'RBS': 'TAG',
    'RP': 'TAG',
    'SYM': 'TAG',
    'TO': 'TAG',
    'UH': 'TAG',
    'VB': 'TAG',
    'VBD': 'TAG',
    'VBG': 'TAG',
    'VBN': 'TAG',
    'VBP': 'TAG',
    'VBZ': 'TAG',
    'WDT': 'TAG',
    'WP': 'TAG',
    'WP$': 'TAG',
    'WRB': 'TAG',
    'XX': 'TAG'}
function_words={'func_w_freq': 'Function words',
    'weren': 'Function words',
    'doesn': 'Function words',
    'does': 'Function words',
    'once': 'Function words',
    'doing': 'Function words',
    'into': 'Function words',
    'nor': 'Function words',
    'don': 'Function words',
    'some': 'Function words',
    'should': 'Function words',
    'at': 'Function words',
    'been': 'Function words',
    'here': 'Function words',
    'you': 'Function words',
    "wouldn't": 'Function words',
    'ain': 'Function words',
    'further': 'Function words',
    'by': 'Function words',
    'what': 'Function words',
    "won't": 'Function words',
    "you've": 'Function words',
    "don't": 'Function words',
    "shan't": 'Function words',
    'no': 'Function words',
    "you'll": 'Function words',
    "shouldn't": 'Function words',
    'they': 'Function words',
    'about': 'Function words',
    'are': 'Function words',
    'herself': 'Function words',
    'hasn': 'Function words',
    'wouldn': 'Function words',
    'again': 'Function words',
    'both': 'Function words',
    'can': 'Function words',
    'hadn': 'Function words',
    'ma': 'Function words',
    "you're": 'Function words',
    'hers': 'Function words',
    "she's": 'Function words',
    'only': 'Function words',
    'her': 'Function words',
    'in': 'Function words',
    'why': 'Function words',
    "hadn't": 'Function words',
    'his': 'Function words',
    'their': 'Function words',
    'which': 'Function words',
    "mustn't": 'Function words',
    'above': 'Function words',
    'its': 'Function words',
    'these': 'Function words',
    'while': 'Function words',
    'over': 'Function words',
    'how': 'Function words',
    'shouldn': 'Function words',
    'so': 'Function words',
    "didn't": 'Function words',
    'has': 'Function words',
    'other': 'Function words',
    'having': 'Function words',
    "hasn't": 'Function words',
    'off': 'Function words',
    'ours': 'Function words',
    'but': 'Function words',
    'out': 'Function words',
    'such': 'Function words',
    're': 'Function words',
    'him': 'Function words',
    'each': 'Function words',
    'not': 'Function words',
    "needn't": 'Function words',
    'we': 'Function words',
    'yourselves': 'Function words',
    'under': 'Function words',
    'from': 'Function words',
    'same': 'Function words',
    'on': 'Function words',
    "isn't": 'Function words',
    "that'll": 'Function words',
    'where': 'Function words',
    'she': 'Function words',
    "should've": 'Function words',
    'aren': 'Function words',
    'will': 'Function words',
    'yours': 'Function words',
    "aren't": 'Function words',
    'itself': 'Function words',
    'most': 'Function words',
    'myself': 'Function words',
    "couldn't": 'Function words',
    'then': 'Function words',
    'themselves': 'Function words',
    "mightn't": 'Function words',
    'shan': 'Function words',
    'against': 'Function words',
    "doesn't": 'Function words',
    'theirs': 'Function words',
    "wasn't": 'Function words',
    'himself': 'Function words',
    'of': 'Function words',
    'up': 'Function words',
    'if': 'Function words',
    'because': 'Function words',
    'were': 'Function words',
    'few': 'Function words',
    'more': 'Function words',
    'wasn': 'Function words',
    'that': 'Function words',
    'the': 'Function words',
    'and': 'Function words',
    'our': 'Function words',
    'after': 'Function words',
    'very': 'Function words',
    'for': 'Function words',
    'my': 'Function words',
    'during': 'Function words',
    'now': 'Function words',
    'me': 'Function words',
    'being': 'Function words',
    'do': 'Function words',
    'isn': 'Function words',
    'before': 'Function words',
    'it': 'Function words',
    'them': 'Function words',
    'to': 'Function words',
    'yourself': 'Function words',
    'll': 'Function words',
    'an': 'Function words',
    'through': 'Function words',
    'all': 'Function words',
    "haven't": 'Function words',
    "weren't": 'Function words',
    'haven': 'Function words',
    'than': 'Function words',
    "it's": 'Function words',
    'had': 'Function words',
    'those': 'Function words',
    'who': 'Function words',
    'this': 'Function words',
    'there': 'Function words',
    'be': 'Function words',
    'as': 'Function words',
    'mustn': 'Function words',
    'any': 'Function words',
    'whom': 'Function words',
    'ourselves': 'Function words',
    'he': 'Function words',
    'needn': 'Function words',
    'your': 'Function words',
    'too': 'Function words',
    'couldn': 'Function words',
    'didn': 'Function words',
    'below': 'Function words',
    'did': 'Function words',
    'am': 'Function words',
    'when': 'Function words',
    'have': 'Function words',
    'mightn': 'Function words',
    'just': 'Function words',
    'between': 'Function words',
    'or': 'Function words',
    've': 'Function words',
    'is': 'Function words',
    'won': 'Function words',
    'until': 'Function words',
    'with': 'Function words',
    'was': 'Function words',
    'down': 'Function words',
    "you'd": 'Function words',
    'own': 'Function words'}
punctuation={"'": 'Punctuation',
    ':': 'Punctuation',
    ',': 'Punctuation',
    '_': 'Punctuation',
    '!': 'Punctuation',
    '?': 'Punctuation',
    ';': 'Punctuation',
    '.': 'Punctuation',
    '"': 'Punctuation',
    '(': 'Punctuation',
    ')': 'Punctuation',
    '-': 'Punctuation',
    '_SP': 'Punctuation',
    "''": 'Punctuation',
    '``': 'Punctuation',
    '$': 'Punctuation'}
structural={'avg_w_len': 'Structural',
    'tot_short_w': 'Structural',
    'tot_digit': 'Structural',
    'tot_upper': 'Structural',
    'avg_s_len': 'Structural',
    'hapax': 'Structural',
    'dis': 'Structural',
    'syllable_count': 'Structural',
    'avg_w_freqc': 'Structural'}
indexes={'yules_K': 'Indexes',
    'shannon_entr': 'Indexes',
    'simposons_ind': 'Indexes',
    'flesh_ease': 'Indexes',
    'flesh_cincade': 'Indexes',
    'dale_call': 'Indexes',
    'gunnin_fox': 'Indexes'}
    
map_features={**letters, **numbers, **ner, **tag, **function_words, **punctuation, **structural, **indexes}


# Simple regression function. Kernel must be one of 'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'
def svr_regressor(x,y, n_fold=10, kernel='rbf'):

    regressor=SVR(kernel=kernel)
    
    kf=KFold(n_splits=n_fold)

    mse_test=[]
    mse_train=[]

    for train_index, test_index in kf.split(x):

        X_train, X_test = x[train_index,:], x[test_index,:]
        Y_train, Y_test = y[train_index], y[test_index]

        regressor.fit(X_train,Y_train)
        train_pred = regressor.predict(X_train)
        test_pred = regressor.predict(X_test)

        # compute train and test MSE
        mse_train.append(mean_squared_error(Y_train, train_pred))
        mse_test.append(mean_squared_error(Y_test, test_pred))

    return np.mean(mse_test), np.mean(mse_train)

# Performs stylistic feature regression from embedding. Return aggregate results by family of features if output=="agg". To get full results, set output=="full" 
def style_embedding_evaluation(embeddings, features, kernel='rbf', n_fold=10, output="agg"):

    features=features.drop(['id'], axis=1).groupby('author').mean().reset_index()

    x=embeddings

    cols=[col for col in features.columns if col != 'author']

    res_dict={}

    for feature in tqdm.tqdm(cols):

        y=np.array(features[feature])

        y=(y - np.mean(y))/np.std(y)
        if np.isnan(y).any():
            continue

        mse_test, mse_train=svr_regressor(x,y, n_fold=n_fold, kernel=kernel)

        res_dict[feature]={"mse_test":mse_test, "mse_train":mse_train}

        res_df=pd.DataFrame.from_dict(res_dict, orient='index')

    if output=="full":
        return res_df.reset_index()
    
    res_df['family']=res_df.index.map(map_features)

    res_df=pd.DataFrame.from_dict({"mean":res_df.groupby("family")["mse_test"].mean(),"std":res_df.groupby("family")["mse_test"].std()}, orient='columns')

    return(res_df)

# Performs style embedding evaluation for multiple embeddings methods. Embeddings should be of shape (n_models, n_authors, embedding_size) and names of size n_models.
# Results are returned aggregated by family of features
def multi_style_evaluation(embeddings, names, features, n_fold=10):

    full_df = pd.DataFrame(columns=['Embedding', 'Function words', 'Indexes', 'NER', 'Punctuation',
    'Structural', 'TAG', 'Letters', 'Numbers'])

    for embedding, name in zip(embeddings, names):

        print(f"Evaluating model : {name}\n", flush=True)
        res_df = style_embedding_evaluation(embedding, features, n_fold=n_fold, output='agg').T

        full_df = full_df.append(res_df.loc['mean'])
        full_df = full_df.append(res_df.loc['std'])

    full_df['Embedding']=np.repeat(names, 2)

    return(full_df)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

# Function to plot spyder chart given the feature regression score by embedding.
def style_spyder_charts(df_results, title="MSE Score for Style Evaluation of Embeddings"):

    df_results=df_results[df_results.index=='mean']

    N = 8
    theta = radar_factory(N, frame='polygon')

    spoke_labels = list(df_results.drop('Embedding', axis=1).columns)

    fig, ax = plt.subplots(figsize=(9, 9),
                            subplot_kw=dict(projection='radar'))

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k'][:len(df_results)]
    
    # Plot the four cases from the example data on separate axes
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1),
                    horizontalalignment='center', verticalalignment='center')
    for d, color in zip(np.array(df_results.loc[:,spoke_labels]), colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    labels = tuple(df_results.Embedding)
    legend = ax.legend(labels, loc=(0.8, -0.1),
                                labelspacing=0.1, fontsize = 'small')

    plt.show()