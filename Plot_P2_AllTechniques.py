"""
====================================================================
Dynamic selection with linear classifiers: P2 Problem
====================================================================
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.preprocessing as preprocessing
import scipy.io as sio
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from deslib.dcs import LCA
from deslib.dcs import MLA
from deslib.dcs import OLA
from deslib.dcs import MCB
from deslib.dcs import Rank
from deslib.des import DESKNN
from deslib.des import KNORAE
from deslib.des import KNORAU
from deslib.des import KNOP
from deslib.des import METADES
from deslib.des import DESFH
from deslib.util.datasets import *
import scipy.io as sio
import pickle


###############################################################################
# Defining helper functions to facilitate plotting the decision boundaries:

def plot_classifier_decision(ax, clf, X, mode='line', **params):
    xx, yy = make_grid(X[:, 0], X[:, 1])
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if mode == 'line':
        ax.contour(xx, yy, Z, **params)
    else:
        ax.contourf(xx, yy, Z, **params)
    ax.set_xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    ax.set_ylim((np.min(X[:, 1]), np.max(X[:, 0])))

def plot_dataset(X, y, ax=None, title=None, **params):
    if ax is None:
        ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25,
               edgecolor='k', **params)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if title is not None:
        ax.set_title(title)

    # P2 Problem
    x = np.arange(0, 1, 0.01)  # start,stop,step
    y1 = (2 * np.sin(x * 10) + 5) / 10
    ax.plot(x, y1, c='k', linewidth=3)
    y2 = ((x * 10 - 2) ** 2 + 1) / 10
    ax.plot(x, y2, c='k', linewidth=3)
    y3 = (-0.1 * (x * 10) ** 2 + 0.6 * np.sin(4 * x * 10) + 8.) / 10.
    ax.plot(x, y3, c='k', linewidth=3)
    y4 = (((x * 10 - 10) ** 2) / 2 + 7.902) / 10.
    ax.plot(x, y4, c='k', linewidth=3)
    # Circle
    # circle = patches.Circle((0.5, 0.5), 0.4, edgecolor = 'black', linestyle = 'dotted', linewidth = '2',facecolor='none')
    # ax.add_patch(circle)

    return ax

def make_grid(x, y, h=.01):
    x_min, x_max = x.min() - 0, x.max() + 0
    y_min, y_max = y.min() - 0, y.max() + 0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

# Prepare the DS techniques. Changing k value to 7.
def initialize_ds(pool_classifiers, X, y, k=7):  # X_DSEL , y_DSEL
    knorau = KNORAU(pool_classifiers, k=k)
    kne = KNORAE(pool_classifiers, k=k)
    desknn = DESKNN(pool_classifiers, k=k)
    ola = OLA(pool_classifiers, k=k)
    lca = LCA(pool_classifiers, k=k)
    mla = MLA(pool_classifiers, k=k)
    mcb = MCB(pool_classifiers, k=k)
    # rank = Rank(pool_classifiers, k=k)
    knop = KNOP(pool_classifiers, k=k)
    meta = METADES(pool_classifiers, k=k)
    desfh = DESFH(pool_classifiers, k=k, theta=theta, gama=NO_Hyperbox_Thereshold)

    list_ds = [knorau, kne, ola, lca, mla, desknn, mcb, knop, meta, desfh]
    names = ['KNORA-U', 'KNORA-E', 'OLA', 'LCA', 'MLA', 'DESKNN', 'MCB', 'KNOP', 'META-DES', 'DES-FH']

    #    list_ds = [knorau, kne, ola, lca, mla, desknn, mcb, rank, knop, meta]
    #    names = ['KNORA-U', 'KNORA-E', 'OLA', 'LCA', 'MLA', 'DESKNN', 'MCB',
    #             'RANK', 'KNOP', 'META-DES']
    # fit the ds techniques
    for ds in list_ds:
        ds.fit(X, y)

    return list_ds, names


# %% Parameters

theta = .1
NO_Hyperbox_Thereshold = 0.9
classifiers_max_depth = 3
NO_classifiers = 100
no_itr = 20

# %% ###############################################################################
# Generating the dataset and training the pool of classifiers.
ran = np.random.randint(1, 10000, 1)
print("RandomState: ", ran)
rng = np.random.RandomState(ran)
# X, y = make_circle_square([1000,1000], random_state=rng)
# X, y = make_banana2(1000, random_state=rng)
# X, y = make_xor(1000, random_state=rng)
X, y = X, y = make_P2([1000, 1000], random_state=rng)

# X = preprocessing.MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=rng)
X_DSEL, X_test, y_DSEL, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=rng)

# model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5 ))
model = CalibratedClassifierCV(Perceptron(max_iter=1000, tol=0.0002))
# model = DecisionTreeClassifier(max_depth=classifiers_max_depth)

pool_classifiers = BaggingClassifier(model, n_estimators=NO_classifiers, random_state=rng)
pool_classifiers.fit(X_DSEL, y_DSEL)

###############################################################################

list_ds, names = initialize_ds(pool_classifiers, X_DSEL, y_DSEL, k=7)

# figB,subB = plt.subplots(1,figsize=(10,10))
# plot_boxes(X,y,list_ds[9].HBoxes)
# plt.show()

#########################################################

fig2, sub2 = plt.subplots(4, 3, figsize=(21, 28))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

ax_data = sub2.flatten()[0]
ax_bagging = sub2.flatten()[1]

plot_dataset(X_DSEL, y_DSEL, ax=ax_data)
ax_data.set_xlim((np.min(X_DSEL[:, 0]), np.max(X_DSEL[:, 0])))
ax_data.set_ylim((np.min(X_DSEL[:, 1]), np.max(X_DSEL[:, 1])))
ax_data.set_title("DSEL Data")

plot_dataset(X_DSEL, y_DSEL, ax=ax_bagging)
ax_data.set_xlim((np.min(X_DSEL[:, 0]), np.max(X_DSEL[:, 0])))
ax_data.set_ylim((np.min(X_DSEL[:, 1]), np.max(X_DSEL[:, 1])))
plot_classifier_decision(ax_bagging, pool_classifiers,
                         X_test, mode='filled', alpha=0.4)

# Plotting the decision border of the DS methods
for ds, name, ax in zip(list_ds, names, sub2.flatten()[2:]):
    plot_dataset(X_DSEL, y_DSEL, ax=ax)
    plot_classifier_decision(ax, ds, X_test, mode='filled', alpha=0.4)
    ax.set_xlim((np.min(X_DSEL[:, 0]) - 0, np.max(X_DSEL[:, 0] + 0)))
    ax.set_ylim((np.min(X_DSEL[:, 1]) - 0, np.max(X_DSEL[:, 1] + 0)))
    ax.set_title(name + " - DSEL")
plt.show()
plt.tight_layout()

###############################################################################
# Evaluation on the test set
# --------------------------
#
# Finally, let's evaluate the classification accuracy of DS techniques and
# Bagging on the test set:

for ds, name in zip(list_ds, names):
    acc = str(ds.score(X_test, y_test))
    print('Accuracy ' + name + ': ' + acc)

acc = str(pool_classifiers.score(X_test, y_test))
print('Accuracy Bagging: ' + acc)