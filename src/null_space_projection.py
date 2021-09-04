#inspired and abstracted from: https://github.com/shauli-ravfogel/nullspace_projection

import pickle
import numpy as np
from pathlib import Path

# sklearn stuff
import sklearn
from sklearn import model_selection
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer





import json
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models import KeyedVectors
import numpy as np
import random
import sklearn
from sklearn import model_selection
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
# from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

import scipy
from scipy import linalg
from scipy import sparse
from scipy.stats.stats import pearsonr
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, SGDRegressor, Perceptron, LogisticRegression

matplotlib.rcParams['agg.path.chunksize'] = 10000

import warnings
warnings.filterwarnings("ignore")

import pickle
from collections import defaultdict, Counter
from typing import List, Dict

import torch
from torch import utils

# import pytorch_lightning as pl
# from pytorch_lightning import Trainer
import copy
import pandas as pd
from gensim.models import FastText
import time
from gensim.scripts.glove2word2vec import glove2word2vec


data_location = [Path('../datasets/bias_in_bios'), '../../../storage/fair_nlp_dataset/data/bias_in_bios']

for d in data_location:
    try:
        train, dev, test = pickle.load(open(d/Path('train.pickle'), 'rb')),\
                                          pickle.load(open(d/Path('dev.pickle'), 'rb')),\
                                          pickle.load(open(d/Path('test.pickle'), 'rb'))

        # train_cls, dev_cls, test_cls = np.load(d/Path('train.pickle_bert_cls.npy')), \
        #                                   np.load(d/Path('dev.pickle_bert_cls.npy')), \
        #                                   np.load(d/Path('test.pickle_bert_cls.npy'))

        train_cls, dev_cls, test_cls = np.load(d/Path('baseline_model_classifier_hidden_train.npy')), \
                                          np.load(d/Path('baseline_model_classifier_hidden_dev.npy')), \
                                          np.load(d/Path('baseline_model_classifier_hidden_test.npy'))
        break
    except FileNotFoundError:
        continue


all_profession = list(set([t['p'] for t in train]))
profession_to_id = {profession: index for index, profession in enumerate(all_profession)}

train_y = [profession_to_id[t['p']] for t in train]
test_y = [profession_to_id[t['p']] for t in test]
dev_y = [profession_to_id[t['p']] for t in dev]

# gender
gender = {'f':0, 'm':1}
train_s = [gender[t['g']] for t in train]
test_s = [gender[t['g']] for t in test]
dev_s = [gender[t['g']] for t in dev]


number_of_labels = len(np.unique(train_y))

train_X, train_y, train_s = train_cls, train_y, train_s
dev_X, dev_y, dev_s = dev_cls, dev_y, dev_s
test_X, test_y, test_s = test_cls, test_y, test_s



import numpy as np


# an abstract class for linear classifiers

class Classifier(object):

    def __init__(self):

        pass

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        """

        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set
        """
        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        raise NotImplementedError




class SKlearnClassifier(Classifier):

    def __init__(self, m):

        self.model = m

    def train_network(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:

        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set / Person's R in the case of regression
        """

        self.model.fit(X_train, Y_train)
        score = self.model.score(X_dev, Y_dev)
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.coef_
        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)

        return w



# debias file
import time
from typing import Dict
import numpy as np
import scipy
from typing import List
from tqdm import tqdm
import random
import warnings
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression

def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T) # orthogonal basis

    P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace

    return P_W

def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param dim: input dim
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis = 0)
    P = I - get_rowspace_projection(Q)

    return P

def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):
    """
    the goal of this function is to perform INLP on a set of user-provided directiosn (instead of learning those directions).
    :param directions: list of vectors, as numpy arrays.
    :param input_dim: dimensionality of the vectors.
    """

    rowspace_projections = []

    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P


def get_debiasing_projection(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                             is_autoregressive: bool,
                             min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                             Y_dev: np.ndarray, by_class=False, Y_train_main=None,
                             Y_dev_main=None, dropout_rate = 0) -> np.ndarray:
    """
    :param classifier_class: the sklearn classifier class (SVM/Perceptron etc.)
    :param cls_params: a dictionary, containing the params for the sklearn classifier
    :param num_classifiers: number of iterations (equivalent to number of dimensions to remove)
    :param input_dim: size of input vectors
    :param is_autoregressive: whether to train the ith classiifer on the data projected to the nullsapces of w1,...,wi-1
    :param min_accuracy: above this threshold, ignore the learned classifier
    :param X_train: ndarray, training vectors
    :param Y_train: ndarray, training labels (protected attributes)
    :param X_dev: ndarray, eval vectors
    :param Y_dev: ndarray, eval labels (protected attributes)
    :param by_class: if true, at each iteration sample one main-task label, and extract the protected attribute only from vectors from this class
    :param T_train_main: ndarray, main-task train labels
    :param Y_dev_main: ndarray, main-task eval labels
    :param dropout_rate: float, default: 0 (note: not recommended to be used with autoregressive=True)
    :return: P, the debiasing projection; rowspace_projections, the list of all rowspace projection; Ws, the list of all calssifiers.
    """
    if dropout_rate > 0 and is_autoregressive:
        warnings.warn("Note: when using dropout with autoregressive training, the property w_i.dot(w_(i+1)) = 0 no longer holds.")

    I = np.eye(input_dim)

    if by_class:
        if ((Y_train_main is None) or (Y_dev_main is None)):
            raise Exception("Need main-task labels for by-class training.")
        main_task_labels = list(set(Y_train_main.tolist()))

    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []

    pbar = tqdm(range(num_classifiers))
    for i in pbar:

        clf = SKlearnClassifier(classifier_class(**cls_params))
        dropout_scale = 1./(1 - dropout_rate + 1e-6)
        dropout_mask = (np.random.rand(*X_train.shape) < (1-dropout_rate)).astype(float) * dropout_scale


        if by_class:
            #cls = np.random.choice(Y_train_main)  # uncomment for frequency-based sampling
            cls = random.choice(main_task_labels)
            relevant_idx_train = Y_train_main == cls
            relevant_idx_dev = Y_dev_main == cls
        else:
            relevant_idx_train = np.ones(X_train_cp.shape[0], dtype=bool)
            relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

        acc = clf.train_network((X_train_cp * dropout_mask)[relevant_idx_train], Y_train[relevant_idx_train], X_dev_cp[relevant_idx_dev], Y_dev[relevant_idx_dev])
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if acc < min_accuracy: continue

        W = clf.get_weights()
        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(W) # projection to W's rowspace
        rowspace_projections.append(P_rowspace_wi)

        if is_autoregressive:

            """
            to ensure numerical stability, explicitly project to the intersection of the nullspaces found so far (instaed of doing X = P_iX,
            which is problematic when w_i is not exactly orthogonal to w_i-1,...,w1, due to e.g inexact argmin calculation).
            """
            # use the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
            # N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))

            P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
            # project

            X_train_cp = (P.dot(X_train.T)).T
            X_dev_cp = (P.dot(X_dev.T)).T

    """
    calculae final projection matrix P=PnPn-1....P2P1
    since w_i.dot(w_i-1) = 0, P2P1 = I - P1 - P2 (proof in the paper); this is more stable.
    by induction, PnPn-1....P2P1 = I - (P1+..+PN). We will use instead Ben-Israel's formula to increase stability and also generalize to the non-orthogonal case (e.g. with dropout),
    i.e., we explicitly project to intersection of all nullspaces (this is not critical at this point; I-(P1+...+PN) is roughly as accurate as this provided no dropout & regularization)
    """

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P, rowspace_projections, Ws





x_train = train_X
x_dev = dev_X
x_test = test_X




y_train = np.asarray(train_y)
y_dev = np.asarray(dev_y)
y_test = np.asarray(test_y)

random.seed(0)
np.random.seed(0)

clf = LogisticRegression(warm_start=True, penalty='l2',
                         solver="saga", multi_class='multinomial', fit_intercept=False,
                         verbose=5, n_jobs=90, random_state=1, max_iter=7)

# params = {'}
# clf = SGDClassifier(loss= 'hinge', max_iter = 4000, fit_intercept= True, class_weight= None, n_jobs= 100)


start = time.time()
idx = np.random.rand(x_train.shape[0]) < 1.0
clf.fit(x_train[idx], y_train[idx])
print("time: {}".format(time.time() - start))
print(clf.score(x_test, y_test))
print(clf.score(x_train, y_train))

import copy
clf_original = copy.deepcopy(clf)

MLP = False


def get_projection_matrix(num_clfs, X_train, Y_train_gender, X_dev, Y_dev_gender, Y_train_task, Y_dev_task, dim):
    is_autoregressive = True
    min_acc = 0.
    # noise = False
    # dim = 768
    dim = 25
    n = num_clfs
    # random_subset = 1.0
    start = time.time()
    TYPE = "svm"

    if MLP:
        x_train_gender = np.matmul(x_train, clf.coefs_[0]) + clf.intercepts_[0]
        x_dev_gender = np.matmul(x_dev, clf.coefs_[0]) + clf.intercepts_[0]
    else:
        x_train_gender = x_train.copy()
        x_dev_gender = x_dev.copy()

    if TYPE == "sgd":
        gender_clf = SGDClassifier
        params = {'loss': 'hinge', 'penalty': 'l2', 'fit_intercept': False, 'class_weight': None, 'n_jobs': 32}
    else:
        gender_clf = LinearSVC
        params = {'penalty': 'l2', 'C': 0.01, 'fit_intercept': True, 'class_weight': None, "dual": False}

    P, rowspace_projections, Ws = get_debiasing_projection(gender_clf, params, n, dim, is_autoregressive, min_acc,
                                                           X_train, Y_train_gender, X_dev, Y_dev_gender,
                                                           Y_train_main=Y_train_task, Y_dev_main=Y_dev_task,
                                                           by_class=True)
    print("time: {}".format(time.time() - start))
    return P, rowspace_projections, Ws


num_clfs = 20
y_dev_gender = np.array(dev_s)
y_train_gender = np.array(train_s)
idx = np.random.rand(x_train.shape[0]) < 1.
P, rowspace_projections, Ws = get_projection_matrix(num_clfs, x_train[idx], y_train_gender[idx], x_dev, y_dev_gender,
                                                    y_train, y_dev, 300)


def get_TPR(y_pred, y_true, p2i, i2p, gender):
    scores = defaultdict(Counter)
    prof_count_total = defaultdict(Counter)

    for y_hat, y, g in zip(y_pred, y_true, gender):

        if y == y_hat:
            scores[i2p[y]][g] += 1

        prof_count_total[i2p[y]][g] += 1

    tprs = defaultdict(dict)
    tprs_change = dict()
    tprs_ratio = []

    for profession, scores_dict in scores.items():
        good_m, good_f = scores_dict["m"], scores_dict["f"]
        prof_total_f = prof_count_total[profession]["f"]
        prof_total_m = prof_count_total[profession]["m"]
        tpr_m = (good_m) / prof_total_m
        tpr_f = (good_f) / prof_total_f

        tprs[profession]["m"] = tpr_m
        tprs[profession]["f"] = tpr_f
        tprs_ratio.append(0)
        tprs_change[profession] = tpr_f - tpr_m

    return tprs, tprs_change, np.mean(np.abs(tprs_ratio))


def get_FPR2(y_pred, y_true, p2i, i2p, y_gender):
    fp = defaultdict(Counter)
    neg_count_total = defaultdict(Counter)
    pos_count_total = defaultdict(Counter)

    label_set = set(y_true)
    # count false positive per gender & class

    for y_hat, y, g in zip(y_pred, y_true, y_gender):

        if y != y_hat:
            fp[y_hat][g] += 1  # count false positives for y_hat

    # count total falses per gender (conditioned on class)

    total_prof_g = defaultdict(Counter)

    # collect POSITIVES for each profession and gender

    for y, g in zip(y_true, y_gender):
        total_prof_g[y][g] += 1

    total_m = sum([total_prof_g[y]["m"] for y in label_set])
    total_f = sum([total_prof_g[y]["f"] for y in label_set])

    # calculate NEGATIVES for each profession and gender

    total_false_prof_g = defaultdict(Counter)
    for y in label_set:
        total_false_prof_g[y]["m"] = total_m - total_prof_g[y]["m"]
        total_false_prof_g[y]["f"] = total_f - total_prof_g[y]["f"]

    fprs = defaultdict(dict)
    fprs_diff = dict()

    for profession, false_pred_dict in fp.items():
        false_male, false_female = false_pred_dict["m"], false_pred_dict["f"]
        prof_total_false_for_male = total_false_prof_g[profession]["m"]
        prof_total_false_for_female = total_false_prof_g[profession]["f"]

        ftr_m = false_male / prof_total_false_for_male
        ftr_f = false_female / prof_total_false_for_female
        fprs[i2p[profession]]["m"] = ftr_m
        fprs[i2p[profession]]["f"] = ftr_f
        fprs_diff[i2p[profession]] = ftr_m - ftr_f

    return fprs, fprs_diff


def similarity_vs_tpr(tprs, word2vec, title, measure, prof2fem):
    professions = list(tprs.keys())
    #
    """ 
    sims = dict()
    gender_direction = word2vec["he"] - word2vec["she"]

    for p in professions:
        sim = word2vec.cosine_similarities(word2vec[p], [gender_direction])[0]
        sims[p] = sim
    """
    tpr_lst = [tprs[p] for p in professions]
    sim_lst = [prof2fem[p] for p in professions]

    # professions = [p.replace("_", " ") for p in professions if p in word2vec]

    plt.plot(sim_lst, tpr_lst, marker="o", linestyle="none")
    plt.xlabel("% women", fontsize=13)
    plt.ylabel(r'$GAP_{female,y}^{TPR}$', fontsize=13)
    for p in professions:
        x, y = prof2fem[p], tprs[p]
        plt.annotate(p, (x, y), size=7, color="red")
    plt.ylim(-0.4, 0.55)
    z = np.polyfit(sim_lst, tpr_lst, 1)
    p = np.poly1d(z)
    plt.plot(sim_lst, p(sim_lst), "r--")
    plt.savefig("{}_vs_bias_{}_bert".format(measure, title), dpi=600)
    print("Correlation: {}; p-value: {}".format(*pearsonr(sim_lst, tpr_lst)))
    plt.show()


def rms_diff(tpr_diff):
    return np.sqrt(np.mean(tpr_diff ** 2))


def save_vecs_and_words(vecs, words):
    def to_string(arr):
        return "\t".join([str(x) for x in arr])

    with open("vecs.txt", "w") as f:
        for v in vecs:
            assert len(v) == 300
            f.write(to_string(v) + "\n")

    with open("labels.txt", "w") as f:
        f.write("Profession\n")
        for w in words:
            f.write(w + "\n")




clf = LogisticRegression(warm_start = True, penalty = 'l2',
                         solver = "sag", multi_class = 'multinomial', fit_intercept = True,
                         verbose = 10, max_iter = 3, n_jobs = 64, random_state = 1)
#clf = SGDClassifier()
P_rowspace = np.eye(25) - P
mean_gender_vec = np.mean(P_rowspace.dot(x_train.T).T, axis = 0)
# 2
print(clf.fit((P.dot(x_train.T)).T, y_train))
#print(clf.fit((x_train.T).T + mean_gender_vec, y_train))



print(f"first acc to: {clf.score((P.dot(x_test.T)).T, y_test)}")


p2i = {profession: index for index, profession in enumerate(all_profession)}
i2p = {value:key for key, value in p2i.items()}

y_pred_before = clf_original.predict(x_test)
test_gender = [d["g"] for d in test]
tprs_before, tprs_change_before, mean_ratio_before = get_TPR(y_pred_before, y_test, p2i, i2p, test_gender)
# similarity_vs_tpr(tprs_change_before, None, "before", "TPR", prof2fem)


y_pred_after = clf.predict((P.dot(x_test.T)).T)
# y_pred_after = clf.predict(X_test)
tprs, tprs_change_after, mean_ratio_after = get_TPR(y_pred_after, y_test, p2i, i2p, test_gender)
# similarity_vs_tpr(tprs_change_after, None, "after", "TPR", prof2fem)


"""     
#print("TPR diff ratio before: {}; after: {}".format(mean_ratio_before, mean_ratio_after))

fprs_before, fprs_change_before = get_FPR2(y_pred_before, y_dev, p2i, i2p, test_gender)
similarity_vs_tpr(fprs_change_before, None, "before", "FPR", prof2fem)


fprs, fprs_change_after = get_FPR2(y_pred_after, y_dev, p2i, i2p, test_gender)
similarity_vs_tpr(fprs_change_after, None, "after", "FPR", prof2fem)

#print("TPR diff ratio before: {}; after: {}".format(mean_ratio_before, mean_ratio_after))
"""
change_vals_before = np.array(list((tprs_change_before.values())))
change_vals_after = np.array(list(tprs_change_after.values()))

print("rms-diff before: {}; rms-diff after: {}".format(rms_diff(change_vals_before), rms_diff(change_vals_after)))






