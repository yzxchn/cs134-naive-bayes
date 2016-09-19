# -*- mode: Python; coding: utf-8 -*-
from collections import Counter
from classifier import Classifier
import math

class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""
    def __init__(self, model={}):
        super(NaiveBayes, self).__init__(model=model)

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)

    def train(self, training_set):
        self.model = {}
        class_count = Counter()
        featureset = set()
        for d in training_set:
            if d.label not in self.model:
                self.model[d.label] = Counter()
            class_count[d.label] += 1
            for f in set(d.features()):
                featureset.add(f)
                self.model[d.label][f] += 1

        labels = self.model.keys()

        self.model["*-PRIORS-*"] = {}
        self.model["*-VOCAB-*"] = featureset
        self.model["*-CLSCOUNT-*"] = class_count

        for l in labels:
            print("{0} {1}".format(l, class_count[l]))
            l_prior = math.log(1.0*class_count[l] / len(training_set))
            self.model["*-PRIORS-*"][l] = l_prior
            for f in self.model[l].keys():
                self.model[l][f] = math.log(1.0*(self.model[l][f] + 1) / (class_count[l] + len(featureset)))


    def classify(self, doc):
        print("testing")
        max_prob = 0.0
        max_label = None
        featureset = self.model["*-VOCAB-*"]
        class_count = self.model["*-CLSCOUNT-*"]
        for l in self.model["*-PRIORS-*"].keys():
            prob_sum = self.model["*-PRIORS-*"][l]
            for f in doc.features():
                if f not in self.model[l].keys():
                    prob_sum += math.log(1.0 / (class_count[l] + len(featureset)))
                else:
                    prob_sum += self.model[l][f]
            if prob_sum >= max_prob:
                max_prob = prob_sum
                max_label = l
        return max_label


    def get_label_index(self, dataset):
        labels = {}
        i = 0
        for d in dataset:
            if d.label not in labels:
                labels[d.label] = i
                i += 1
        return labels

c = NaiveBayes()
