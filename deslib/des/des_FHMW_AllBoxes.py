# coding=utf-8

# FH-DES with contraction process
#     1- Check just the nearest hyperbox
#     2- Create a new hyperbox if the nearst hyperboxox has contraction
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.utils import shuffle
from deslib.des.base import BaseDES
from deslib.util.fuzzy_hyperbox import Hyperbox
from matplotlib.patches import Rectangle


class DESFHMW_allboxes(BaseDES):

    def __init__(self, pool_classifiers=None,
                 k=7, DFP=False,
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30,
                 random_state=None,
                 knn_classifier='knn',
                 DSEL_perc=0.5,
                 HyperBoxes=[],
                 theta=0.05,
                 mu=0.991,
                 mis_sample_based = True,
                 doContraction = True,
                 thetaCheck = True,
                 multiCore_process = False,
                 shuffle_dataOrder = False):
        self.theta = theta
        self.mu = mu
        self.mis_sample_based = mis_sample_based
        self.HBoxes = []
        self.NO_hypeboxes = 0
        self.doContraction = doContraction
        self.thetaCheck = thetaCheck
        self.multiCore_process = multiCore_process
        self.shuffle_dataOrder = shuffle_dataOrder
        ############### it should be based on Clustering #############################
        super(DESFHMW_allboxes, self).__init__(pool_classifiers=pool_classifiers,
                                    with_IH=with_IH,
                                    safe_k=safe_k,
                                    IH_rate=IH_rate,
                                    mode='hybrid',  # hybrid,weighting
                                    # mode='weighting',
                                    random_state=random_state,
                                    DSEL_perc=DSEL_perc)

    def fit(self, X, y):

        super(DESFHMW_allboxes, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Theta must be between 0 and 1.")
###
        if self.multiCore_process == False:
            for classifier_index in range(self.n_classifiers_):
                boxes = self.FMM_train(classifier_index)
                self.HBoxes.extend(boxes)
                self.NO_hypeboxes += len(boxes)
        else:
            # classifier_index = range(self.n_classifiers_)
            no_processes = int(multiprocessing.cpu_count() /2)+1
            with multiprocessing.Pool(processes=no_processes) as pool:
                box_list = pool.map(self.FMM_train, range(self.n_classifiers_))
                for boxes in box_list:
                    self.HBoxes.extend(boxes)
                    self.NO_hypeboxes += len(boxes)


    # def remove_bad_labels:
    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):
        boxes_classifier = np.zeros((len(self.HBoxes), 1))
        boxes_W = np.zeros((len(self.HBoxes), self.n_features_))
        boxes_V = np.zeros((len(self.HBoxes), self.n_features_))
        boxes_center = np.zeros((len(self.HBoxes), self.n_features_))
        if self.mis_sample_based:
            competences_ = np.ones([len(query), self.n_classifiers_])
        else:
            competences_ = np.zeros([len(query), self.n_classifiers_])
        for i in range(len(self.HBoxes)):
            boxes_classifier[i] = self.HBoxes[i].clsr
            boxes_W[i] = self.HBoxes[i].Max
            boxes_V[i] = self.HBoxes[i].Min
            boxes_center[i] = (self.HBoxes[i].Max + self.HBoxes[i].Min) / 2
        boxes_W = boxes_W.reshape(self.NO_hypeboxes, 1, self.n_features_)
        boxes_V = boxes_V.reshape(self.NO_hypeboxes, 1, self.n_features_)
        boxes_center = boxes_center.reshape(self.NO_hypeboxes, 1, self.n_features_)
        Xq = query.reshape(1, len(query), self.n_features_)

        ## Membership Calculation
        halfsize = ((boxes_W - boxes_V) / 2).reshape(self.NO_hypeboxes, 1, self.n_features_)
        d = np.abs(boxes_center - Xq) - halfsize
        d[d < 0] = 0
        dd = np.linalg.norm(d, axis=2)
        dd = dd / np.sqrt(self.n_features_)
        m = 1 - dd  # m: membership
        m = np.power(m, 6)

        classifiers, indices, count = np.unique(boxes_classifier, return_counts=True, return_index=True)
        k = 0
        for clsr in classifiers:
            c_range = range(indices[k], indices[k] + count[k])
            k += 1
            clsrBoxes_m = m[c_range]
            if len(c_range) > 1:
                # bb_indexes = np.argpartition(-clsrBoxes_m, kth=2, axis=0)[:2]
                bb_indexes = np.argsort(-clsrBoxes_m, axis=0)
                b1 = bb_indexes[0, :]
                b2 = bb_indexes[1, :]
                for i in range(0, len(query)):
                    if clsrBoxes_m[b1[i], i] == 1:  # if the query sample is located inside or near to the box
                        competences_[i, int(clsr)] = 1
                    else:
                        competences_[i, int(clsr)] = clsrBoxes_m[b1[i], i] * 0.7 + clsrBoxes_m[b2[i], i] * 0.3


            else:  # In case that we have only one hyperbox for the classifier
                for i in range(0, len(query)):
                    competences_[i, int(clsr)] = clsrBoxes_m[0, i]

        #### was mistake ####
        if self.mis_sample_based:
            competences_ = np.max(competences_) - competences_
            # competences_ = np.sqrt(self.n_features_)  - competences_

        scaler = preprocessing.MinMaxScaler()
        competences_ = scaler.fit_transform(competences_)

        return competences_


    def FMM_train(self, classifier_ind):
        #        print(np.size(samples_ind))
        negetive_boxes = []
        positive_boxes = []
        X = self.DSEL_data_
        y = self.DSEL_processed_[:,classifier_ind]

        ############################################################# Shuffle
        if self.shuffle_dataOrder:
            X, y = shuffle(X,y,random_state = classifier_ind)

        for ind, x in enumerate(X):
            ############## Type: Miss or Correct Classified ############
            if y[ind] == False: # miss classified sample
                activeBoxes = negetive_boxes
                conBoxes = positive_boxes
                box_type = 'M'

            else:
                activeBoxes = positive_boxes
                conBoxes = negetive_boxes
                box_type = 'C'

            #############################################################
            # Creation first box
            if len(activeBoxes) < 1:
                # Create the first Box
                newBox = Hyperbox(v=x, w=x, classifier=classifier_ind, theta=self.theta, type=box_type)
                newBox.contract(conBoxes)
                activeBoxes.append(newBox)
                continue

            # X is in a box?
            IsInBox = False
            for box in activeBoxes:
                if np.all(box.Min < X) and np.all(box.Max > X):
                # if box.membership(X) > .95:
                    IsInBox = True
                    break
            if IsInBox:
                # nop
                continue
            ######################## Expand ############################
            # Sort boxes by its distance 
            expanded = False
            dst_list = np.zeros((len(activeBoxes),))
            for i, box in enumerate(activeBoxes):
                dst_list[i] = np.linalg.norm(X - box.Center)
            sorted_indexes = np.argsort(dst_list)[::-1]
            for ind in sorted_indexes:
                nearest_box = activeBoxes[ind]
                if (nearest_box.is_expandable(x)):
                    nearest_box.expand(x)
                    nearest_box.contract(conBoxes)
                    expanded = True
                    break

            # # Finding nearest box
            # nDist = -np.inf
            # hi_mem_box = None
            # for box in activeBoxes:
            #     mem = box.membership(x)
            #     if mem > nDist:
            #         hi_mem_box = box
            #         nDist = mem
            # if (hi_mem_box.is_expandable(x)):
            #     hi_mem_box.expand(x)
            #     hi_mem_box.contract(conBoxes)
            #     continue

            ######################## Creation ############################
            #            else:
            if expanded == False:
                newBox = Hyperbox(v=x, w=x, classifier=classifier_ind, theta=self.theta, type=box_type)
                newBox.contract(conBoxes)
                activeBoxes.append(newBox)

        # self.HBoxes.extend(positive_boxes)
        return negetive_boxes

    def select(self, competences):

        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        max_value = np.max(competences, axis=1)
        selected_classifiers = (
                competences >= self.mu * max_value.reshape(competences.shape[0], -1))


        return selected_classifiers

