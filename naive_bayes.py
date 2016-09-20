# -*- mode: Python; coding: utf-8 -*-
from collections import Counter
from classifier import Classifier
import numpy as np
import math

class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""
    def __init__(self, model={}):
        super(NaiveBayes, self).__init__(model=model)

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)

    def train(self, train_set):
        count_table = {}
        label_to_index = self.get_label_index(train_set)
        label_count = np.zeros(len(label_to_index))
        for doc in train_set:
            l = doc.label
            if len(l) > 0:
                label_count[label_to_index[l]] += 1
                for feature in set(doc.features()):
                    if feature not in count_table:
                        count_table[feature] = np.ones(len(label_to_index))
                    count_table[feature][label_to_index[l]] += 1

        for feature in count_table.keys():
            for label in label_to_index.keys():
                count_table[feature][label_to_index[label]] =  math.log(count_table[feature][label_to_index[label]] / \
                                                                        (label_count[label_to_index[label]] + 2.0))
        count_table["*-UNK-*"] = np.zeros(len(label_to_index))
        for label in label_to_index.keys():
            count_table["*-UNK-*"][label_to_index[label]] = math.log(1.0 / (label_count[label_to_index[label]] + 2.0))

        priors = np.log(label_count / (1.0*len(train_set)))
        cond_prob_table = count_table
        feature_set = count_table.keys()

        self.model = (cond_prob_table, priors, label_to_index, feature_set, label_count)

    def classify(self, instance):
        cond_prob_table, priors, label_to_index, feature_set, label_count = self.model
        results = []
        for label in label_to_index.keys():
            prob_sum = priors[label_to_index[label]]
            for feature in set(instance.features()):
                if feature not in feature_set:
                    prob_sum += cond_prob_table["*-UNK-*"][label_to_index[label]]
                else:
                    prob_sum += cond_prob_table[feature][label_to_index[label]]
            results.append((label, prob_sum))
        return max(results, key=lambda (x, y): y)[0]

    def get_label_index(self, dataset):
        labels = {}
        i = 0
        for d in dataset:
            if d.label not in labels and len(d.label) > 0:
                labels[d.label] = i
                i += 1
        return labels

