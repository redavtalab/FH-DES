# -*- coding: utf-8 -*-

import numpy as np


class Hyperbox:
    def __init__(self, v=0, w=0, classifier=0, theta=.1, type=None):
        self.Min = v
        self.Max = w
        self.clsr = classifier
        self.Center = (v + w) / 2
        self.type = type
        # self.wCenter =(v+w)/2
        self.theta = theta
        # self.samples=[]
        # self.add_sample(v)

    def expand(self, x):
        self.Min = np.minimum(self.Min, x)
        self.Max = np.maximum(self.Max, x)
        self.Center = (self.Min + self.Max) / 2
        # self.add_sample(x)

    def is_overlapped(self, box):
        minW = np.minimum(self.Max, box.Max)
        maxV = np.maximum(self.Min, box.Min)
        return all(maxV < minW)

    def will_overlap_accur(self, conBoxes, x):
        candidV = np.minimum(self.Min, x)
        candidW = np.maximum(self.Max, x)
        for box in conBoxes:
            minW = np.minimum(candidW, box.Max)
            maxV = np.maximum(candidV, box.Min)
            if all(maxV < minW):
                return True
        return False

    def contract(self, conBoxes):
        ndimension = len(self.Min)
        for box in conBoxes:
            if self.type == box.type:
                print("Type of boxes are the same")
                continue
            if not self.is_overlapped(box):
                continue
            minOverlap = np.inf
            dimOverlap = -1
            for n in range(ndimension):
                if (self.Min[n] <= box.Min[n] and box.Min[n] < self.Max[n] and self.Max[n] <= box.Max[n]):
                    if (self.Max[n] - box.Min[n]) < minOverlap:
                        minOverlap = self.Max[n] - box.Min[n]
                        type = 1
                        dimOverlap = n

                elif box.Min[n] <= self.Min[n] and self.Min[n] < box.Max[n] and box.Max[n] <= self.Max[n]:
                    if (box.Max[n] - self.Min[n]) < minOverlap:
                        minOverlap = box.Max[n] - self.Min[n]
                        type = 2
                        dimOverlap = n



                elif self.Min[n] <= box.Min[n] and box.Max[n] <= self.Max[n]:
                    m = min((box.Min[n] - self.Min[n]), (self.Max[n] - box.Max[n]))
                    if m < minOverlap:
                        minOverlap = m
                        type = 3
                        dimOverlap = n

                elif box.Min[n] <= self.Min[n] and self.Max[n] <= box.Max[n]:
                    m = min((self.Min[n] - box.Min[n]), (box.Max[n] - self.Max[n]))
                    if m < minOverlap:
                        minOverlap = m
                        type = 4
                        dimOverlap = n

            if type == 1:
                box.Min[dimOverlap] = (self.Max[dimOverlap] + box.Min[dimOverlap]) / 2
                self.Max[dimOverlap] = box.Min[dimOverlap]

            elif type == 2:
                self.Min[dimOverlap] = (box.Max[dimOverlap] + self.Min[dimOverlap]) / 2
                box.Max[dimOverlap] = self.Min[dimOverlap]

            elif type == 3:
                if (box.Max[dimOverlap] - self.Min[dimOverlap]) < (self.Max[dimOverlap] - box.Min[dimOverlap]):
                    self.Min[dimOverlap] = box.Max[dimOverlap]
                else:
                    self.Max[dimOverlap] = box.Min[dimOverlap]

            else:
                if (self.Max[dimOverlap] - box.Min[dimOverlap]) < (box.Max[dimOverlap] - self.Min[dimOverlap]):
                    box.Min[dimOverlap] = self.Max[dimOverlap]
                else:
                    box.Max[dimOverlap] = self.Min[dimOverlap]

    def is_expandable(self, x, theta=-1):
        if theta == -1:
            theta = self.theta
        candidV = np.minimum(self.Min, x)
        candidW = np.maximum(self.Max, x)

        return all((candidW - candidV) < theta)

    def will_exceed_samples(self, x, con_samples):
        candidV = np.minimum(self.Min, x)
        candidW = np.maximum(self.Max, x)
        no_features = len(self.Min)
        V = candidV.reshape(1, no_features)
        W = candidW.reshape(1, no_features)

        return np.any(np.all(V <= con_samples, 1) & np.all(W >= con_samples, 1))

    def contract_samplesBased(self, con_samples):
        di = 0
        v = self.Min
        w = self.Max
        mi = con_samples - v
        ma = w - con_samples

        inds1 = np.all(mi > 0, 1)
        inds2 = np.all(ma > 0, 1)
        confilicts_inds = np.where(inds1 & inds2)

        for ind in list(confilicts_inds[0]):
            if np.all(v < con_samples[ind]) and np.all(w > con_samples[ind]):
                mi = con_samples[ind] - self.Min
                ma = self.Max - con_samples[ind]
                if min(mi) < min(ma):
                    d = np.where(mi == min(mi))
                    self.Min[d] = con_samples[ind, d]
                else:
                    d = np.where(ma == min(ma))
                    self.Max[d] = con_samples[ind, d]

    def membership(self, x):

        disvec = np.abs(self.Center - x)
        halfsize = (self.Max - self.Min) / 2
        d = disvec - halfsize
        m = np.linalg.norm(d[d > 0])
        m = m / np.sqrt(len(x))  # adapting with high dimensional problems
        m = 1 - m
        m = np.power(m, 4)
        return m

        # d = np.linalg.norm(x-self.Center)
        # po= d/np.sqrt(len(x)-1)  # adapting with high dimensional problems
        # m = np.power(0.05,po)

    #         t = np.sqrt(len(x))/3
    #         m = (1-d/t)
    #         if m<0:
    #             m=0
    #
    #        m = 1 - np.sqrt(np.sum((x-self.wCenter)**2))

    # y = 2
    # m = np.inf
    # ndimension = np.size(x)
    # for n in range(ndimension):
    #    m = np.minimum(m, np.minimum(1- self.f(x[n] - self.Max[n] , y) , 1-self.f(self.Min[n] - x[n],y)))
    #       m = 1-m
    #
    #        return m

    # def add_sample(self,x):
    #    self.samples.append(x)
    #        self.wCenter = ((len(self.samples)-1) * self.wCenter + x) / len(self.samples)

    def f(self, r, y):
        if r * y > 1:
            return 1
        if r * y >= 0:
            return r * y
        return 0
#