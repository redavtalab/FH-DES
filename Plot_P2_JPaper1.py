"""
====================================================================
Dynamic selection with linear classifiers: P2 Problem
====================================================================

"""
# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.patches as patches

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
import sklearn.preprocessing as preprocessing
import scipy.io as sio
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from deslib.static.oracle import Oracle
# from deslib.dcs import LCA
# from deslib.dcs import MLA
from deslib.dcs import OLA
# from deslib.dcs import MCB
# from deslib.dcs import Rank
#
# from deslib.des import DESKNN
# from deslib.des import KNORAE
# from deslib.des import KNORAU
# from deslib.des import KNOP
# from deslib.des import METADES
from deslib.des import FHDES_JFB,FHDES_Allboxes, FHDES_prior,DESFHMW_JFB, DESFHMW_allboxes,DESFHMW_prior,FHDES_Allboxes_vector
from deslib.util.datasets import *

###############################################################################
def plot_classifier_decision(ax, clf, X, mode='line', **params):
    xx, yy = make_grid(X[:, 0], X[:, 1])
    #    xx = preprocessing.MinMaxScaler().fit_transform(xx)
    #    yy = preprocessing.MinMaxScaler().fit_transform(yy)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if mode == 'line':
        ax.contour(xx, yy, Z, **params)
    else:
        ax.contourf(xx, yy, Z, **params)
    ax.set_xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    ax.set_ylim((np.min(X[:, 1]), np.max(X[:, 0])))

def plot_dataset(X, y, ax=None, title=None, boxes=None, class_num=0, **params):
    if ax is None:
        ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=15,
               edgecolor='k', **params)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if title is not None:
        ax.set_title(title)

    # P2 Problem
    x = np.arange(0, 1, 0.01)  # start,stop,step
    y1 = (2 * np.sin(x * 10) + 5) / 10
    ax.plot(x, y1,'g')
    y2 = ((x * 10 - 2) ** 2 + 1) / 10
    ax.plot(x, y2,'g')
    y3 = (-0.1 * (x * 10) ** 2 + 0.6 * np.sin(4 * x * 10) + 8.) / 10.
    ax.plot(x, y3,'g')
    y4 = (((x * 10 - 10) ** 2) / 2 + 7.902) / 10.
    ax.plot(x, y4,'g')
    # Circle
    # circle = patches.Circle((0.5, 0.5), 0.4, edgecolor = 'black', linestyle = 'dotted', linewidth = '2',facecolor='none')
    # ax.add_patch(circle)
    if boxes != None:
        if ax is None:
            ax = plt.gca()

        for bb in boxes:
            [hei, wid] = bb.Max - bb.Min
            if bb.clsr != class_num:
                continue
            rect = Rectangle(bb.Min , hei , wid , linewidth=1, edgecolor='r',
                             facecolor='none')  # fill=False
            ax.add_patch(rect)

        #     pc = PatchCollection(hboxes,facecolor=facecolor,alpha=alpha,edgecolor=edgecolor)
        #     ax.add_collection(pc)

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('HBoxes')

    return ax

def make_grid(x, y, h=.02):
    x_min, x_max = x.min() - 0, x.max() + 0
    y_min, y_max = y.min() - 0, y.max() + 0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

########################################## plot boxes
def plot_boxes(X, L, boxes, ax=None):
    if ax is None:
        ax = plt.gca()
    #        ax.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25,
    #        edgecolor='k', **params)
    #        clf, gscatter(X[:,1] , X[:,2] , L )
    c = ['b', 'g', 'k', 'r', 'y', 'c', 'm', 'w']

    for bb in boxes:
        [hei, wid] = bb.Max - bb.Min

        rect = Rectangle(bb.Min, hei , wid , linewidth=1, edgecolor=c[bb.clsr],
                         facecolor='none')  # fill=False
        ax.add_patch(rect)

    #     pc = PatchCollection(hboxes,facecolor=facecolor,alpha=alpha,edgecolor=edgecolor)
    #     ax.add_collection(pc)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('HBoxes')
    return ax

def initialize_ds(pool_classifiers, XDSEL, yDSEL, k=7):  # X_DSEL , y_DSEL

    ola = OLA(pool_classifiers, k=k)
    FH_1 = FHDES_JFB(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True, doContraction=False, thetaCheck=True, multiCore_process=True)
    FH_2 = FHDES_JFB(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True, doContraction=True, thetaCheck=False, multiCore_process=True)

    FH_3 = FHDES_Allboxes(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True, doContraction=False, thetaCheck=True, multiCore_process=True)
    FH_4 = FHDES_Allboxes(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True, doContraction=True, thetaCheck=False, multiCore_process=True)

    FH_5 = FHDES_JFB(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                     doContraction=True, thetaCheck=True, multiCore_process=True)
    FH_6 = DESFHMW_JFB(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                       doContraction=True, thetaCheck=True, multiCore_process=True)

    FH_7 = FHDES_Allboxes(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                          doContraction=True, thetaCheck=True, multiCore_process=True)
    FH_8 = DESFHMW_allboxes(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                            doContraction=True, thetaCheck=True, multiCore_process=True)

    FH_9 = FHDES_prior(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                       doContraction=True, thetaCheck=True, multiCore_process=True)
    FH_10 = DESFHMW_prior(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                          doContraction=True, thetaCheck=True, multiCore_process=True)

    FH_vec = FHDES_Allboxes_vector(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                          doContraction=True, thetaCheck=True, multiCore_process=True)
    oracle = Oracle(pool_classifiers)
    list_ds = [ ola, FH_vec]
    #    list_ds = [knorau, kne, lca, mla, desknn, mcb, rank, knop, meta, desfh]
    names = ['OLA','FH_1']

    # FH_1.fit(XDSEL,yDSEL)
    for ds in list_ds:
        ds.fit(XDSEL, yDSEL)

    return list_ds, names

# %% Parameters

theta = .2
NO_samples = 500
rng = 118

NO_Hyperbox_Thereshold = 0.85
classifiers_max_depth = 3
NO_classifiers = 2
no_itr = 1

# %%
# Generating the dataset and training the pool of classifiers.
#
# ran = np.random.randint(1, 10000, 1)
# print("RandomState: ", ran)

# X, y = make_circle_square([500,500], random_state=rng)
# X, y = make_banana2(1000, random_state=rng)
# X, y = make_xor(1000, random_state=rng)
X_DSE, y_DSE = make_P2([5000,5000], random_state=rng)
X_DSE, y_DSE = shuffle(X_DSE, y_DSE,random_state=rng)
X_DSEL = X_DSE[:NO_samples,:]
y_DSEL = y_DSE[:NO_samples]
X_tt , y_tt = make_P2([1000,1000], random_state=rng)

print(X_DSEL.shape)


# Spliting the Tarin and Test data
X_train, X_test, y_train, y_test = train_test_split(X_tt, y_tt, test_size=0.5, stratify=y_tt,
                                                    random_state=rng)  # stratify=y

# Normalizing the dataset to have 0 mean and unit variance.
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# learner = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5 ))

# learner = DecisionTreeClassifier(max_depth=classifiers_max_depth)
learner = Perceptron(max_iter=100, tol=10e-3, alpha=0.001, penalty=None, random_state=rng)
# model = CalibratedClassifierCV(learner, cv=5, method='isotonic')

pool_classifiers = BaggingClassifier(learner, n_estimators=NO_classifiers, random_state=rng)

pool_classifiers.fit(X_train, y_train)

# %%
###############################################################################

list_ds, names = initialize_ds(pool_classifiers, X_DSEL, y_DSEL, k=7)

# figB, subB = plt.subplots(1, figsize=(10, 10))
# plot_boxes(X, y, list_ds[1].HBoxes)
# plt.show()


boxes=list_ds[1].HBoxes

#
fig, sub = plt.subplots(2, 2, figsize=(20, 20))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

ax_c1 = sub.flatten()[0]
ax_c2 = sub.flatten()[1]
ax_kon = sub.flatten()[2]
ax_fh = sub.flatten()[3]

plot_dataset(X_DSEL, y_DSEL, ax=ax_c1, boxes=list_ds[1].HBoxes ,class_num=0 )
ax_c1.set_xlim((np.min(X_DSEL[:, 0]), np.max(X_DSEL[:, 0])))
ax_c1.set_ylim((np.min(X_DSEL[:, 1]), np.max(X_DSEL[:, 1])))
plot_classifier_decision(ax_c1, pool_classifiers[0],
                         X_test, mode='filled', alpha=0.4)
ax_c1.set_title("C1")

plot_dataset(X_DSEL, y_DSEL, ax=ax_c2,  boxes=list_ds[1].HBoxes ,class_num=1 )
ax_c2.set_xlim((np.min(X_DSEL[:, 0]), np.max(X_DSEL[:, 0])))
ax_c2.set_ylim((np.min(X_DSEL[:, 1]), np.max(X_DSEL[:, 1])))
plot_classifier_decision(ax_c2, pool_classifiers[1],
                         X_test, mode='filled', alpha=0.4)
ax_c2.set_title("C2")

plot_dataset(X_DSEL, y_DSEL, ax=ax_kon)
plot_classifier_decision(ax_kon, list_ds[0],
                         X_test, mode='filled', alpha=0.4)
ax_kon.set_xlim((np.min(X_DSEL[:, 0]), np.max(X_DSEL[:, 0])))
ax_kon.set_ylim((np.min(X_DSEL[:, 1]), np.max(X_DSEL[:, 1])))
ax_kon.set_title(names[0])
xx = [[.8 , .8],[.5 , .5],[.7, .3]]
y1 = pool_classifiers[0].predict(xx)
y2 = pool_classifiers[1].predict(xx)
yy = list_ds[1].predict(xx)
plot_dataset(X_DSEL, y_DSEL, ax=ax_fh)
plot_classifier_decision(ax_fh, list_ds[1],
                         X_test, mode='filled', alpha=0.4)
ax_fh.set_xlim((np.min(X_DSEL[:, 0]), np.max(X_DSEL[:, 0])))
ax_fh.set_ylim((np.min(X_DSEL[:, 1]), np.max(X_DSEL[:, 1])))
ax_fh.set_title(names[1])

plt.show()
plt.tight_layout()
# %%
###############################################################################
# Evaluation on the test set
# --------------------------
#
# Finally, let's evaluate the classification accuracy of DS techniques and
# Bagging on the test set:
print("NO_Hyperboxes is:", len(list_ds[1].HBoxes))

for ds, name in zip(list_ds, names):
    print('Accuracy ' + name + ': ' + str(ds.score(X_test, y_test)))
print('Accuracy Bagging: ' + str(pool_classifiers.score(X_test, y_test)))