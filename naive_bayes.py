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
        # count_table is a dictionary of Counter objects, the keys are the labels of the documents
        count_table = {}
        # This is for counting the total number of documents of each label
        label_count = Counter()
        # The "vocabulary"
        feature_set = set()
        for doc in train_set:
            l = doc.label
            # ignore the blog documents that have an empty unicode object as label.
            if type(l) is unicode and len(l) == 0:
                pass
            else:
                label_count[l] += 1
                if l not in count_table:
                    count_table[l] = Counter()
                for feature in doc.features():
                    feature_set.add(feature)
                    count_table[l][feature] += 1
        # convert counts to likelihoods P(X | Y)
        for label in count_table.keys():
            for feature in count_table[label].keys():
                count_table[label][feature] = math.log((count_table[label][feature] + 1.0) / \
                                                                     (label_count[label] + len(feature_set)))
            # Add an Unknown probability for features not seen in training
            count_table[label]["*-UNK-*"] = math.log(1.0 / (label_count[label] + len(feature_set)))
            # Convert count of documents for each label to prior probabilities P(Y)
            label_count[label] = math.log(label_count[label] / (1.0*len(train_set)))
        # rename the tables
        priors = label_count
        cond_prob_table = count_table

        # package the two tables into model
        self.model = (cond_prob_table, priors)
        # self.save("bernoulli_laplace")

    def classify(self, instance):
        # self.load("bernoulli_laplace")
        cond_prob_table, priors = self.model
        results = []
        for label, likelihoods in cond_prob_table.items():
            prob_sum = priors[label]
            for feature in instance.features():
                if feature not in likelihoods:
                    prob_sum += likelihoods["*-UNK-*"]
                else:
                    prob_sum += likelihoods[feature]
            results.append((label, prob_sum))
        return max(results, key=lambda (x, y): y)[0]

