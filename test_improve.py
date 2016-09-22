from __future__ import division

from corpus import Document, BlogsCorpus
from naive_bayes import NaiveBayes
from test_naive_bayes import accuracy, BagOfWords
import nltk

import sys
from collections import defaultdict, Counter
from random import shuffle, seed
from unittest import TestCase, main, skip


class NoStopWords(Document):
    def features(self):
        tokens = nltk.word_tokenize(self.data)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        no_stop_words = [w.lower() for w in tokens if w not in stop_words]
        return no_stop_words

class NoStopWords_MaxWordLength(Document):
    def features(self):
        tokens = nltk.word_tokenize(self.data)
        stop_words = set(nltk.corpus.stopwords.words())
        max_len = 0
        no_stop_words = []
        for w in tokens:
            if w not in stop_words:
                no_stop_words.append(w)
                if len(w) > max_len:
                    max_len = len(w)
        no_stop_words.append("max_word_len({})".format(max_len))

        return no_stop_words

class PronPrepDet(Document):
    def features(self):
        tokens = nltk.word_tokenize(self.data)
        tagged = nltk.pos_tag(tokens)
        tags = {"PRP", "PRP$", "IN", "RP", ".", "DT"}
        words = [w.lower() for w, t in tagged if t in tags]
        return words

class NoStopWords_PronPrepDet(Document):
    def features(self):
        tokens = nltk.word_tokenize(self.data)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tagged = nltk.pos_tag(tokens)
        tags = {"PRP", "PRP$", "IN", "RP", ".", "DT"}
        words = [w.lower() for w, t in tagged if w not in stop_words]
        words2 = [w.lower() for w, t in tagged if t in tags]
        words.extend(words2)

        return words

class NoStopWords_Pron(Document):
    def features(self):
        tokens = nltk.word_tokenize(self.data)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tagged = nltk.pos_tag(tokens)
        tags = {"PRP", "PRP$", "IN", "RP", ".", "DT"}
        words = [w.lower() for w, t in tagged if w not in stop_words or t in tags]

        return words


class ADVADJVB(Document):
    def features(self):
        tokens = nltk.word_tokenize(self.data)
        tagged = nltk.pos_tag(tokens)
        vb_adj = [w for w, t in tagged if t in {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}]
        return vb_adj


def evaluate(classifier, test, verbose=sys.stderr):
    """Run the classifier on the test data, then display the result"""
    labels = find_labels(test)
    confusion_matrix = defaultdict(Counter)
    for doc in test:
        predict = classifier.classify(doc)
        correct = doc.label
        confusion_matrix[correct][predict] += 1
    result_f1 = []
    #print(confusion_matrix)
    print("{:>10}{:>10}{:>10}{:>20}".format("Label", "Precision", "Recall", "F1-Measure"))
    for l in labels:
        #print(recall(confusion_matrix, l))
        p = precision(confusion_matrix, l)
        r = recall(confusion_matrix, l)
        f1 = f1_measure(p, r)
        result_f1.append(f1)
        print("{:>10}{:>10.2f}{:>10.2f}{:>15.2f}".format(l, p, r, f1))
    print("Accuracy: {:.2f}".format(accuracy_improve(confusion_matrix)))
    print("Macro-averaged F1: {:.2f}".format(1.0*sum(result_f1)/len(result_f1)))


def precision(confusion_matrix, label):
    tp = confusion_matrix[label][label]
    fp = sum(v[label] for k,v in confusion_matrix.items() if k != label)
    return 1.0*tp/(tp+fp)

def recall(confusion_matrix, label):
    tp = confusion_matrix[label][label]
    fn = sum(v for k, v in confusion_matrix[label].items() if k != label)
    return 1.0*tp/(tp+fn)

def f1_measure(pre, rec):
    return 2.0*pre*rec/(pre+rec)

def accuracy_improve(confusion_matrix):
    correct = sum(v[k] for k, v in confusion_matrix.items())
    total = sum(sum(c for l, c in v.items()) for k, v in confusion_matrix.items())
    return 1.0*correct/total


def find_labels(testset):
    labels = set()
    for doc in testset:
        l = doc.label
        # ignore the blog documents that have an empty unicode object as label.
        if type(l) is unicode and len(l) == 0:
            pass
        else:
            labels.add(l)
    return labels



class NaiveBayesImprovementTest(TestCase):
    """Test for improved Naive Bayes Classifier"""

    def split_blogs_corpus(self, document_class):
        """Split the blog post corpus into training and test sets"""
        blogs = BlogsCorpus(document_class=document_class)
        self.assertEqual(len(blogs), 3232)
        seed(hash("blogs"))
        shuffle(blogs)
        return (blogs[:3000], blogs[3000:])

    def test_blogs_improve(self):
        """Classify blog authors"""
        train, test = self.split_blogs_corpus(NoStopWords)
        classifier = NaiveBayes()
        classifier.load("./models/multinomial_laplace_no_stop_words")
        #classifier.train(train)
        #classifier.save("./models/multinomial_laplace_no_stop_words_pron_prep_det")
        evaluate(classifier, test)

    # def split_blogs_corpus_imba(self, document_class):
    #     blogs = BlogsCorpus(document_class=document_class)
    #     imba_blogs = blogs.split_imbalance()
    #     return (imba_blogs[:1600], imba_blogs[1600:])
    #
    # def test_blogs_imba(self):
    #     train, test = self.split_blogs_corpus_imba(NoStopWords)
    #     classifier = NaiveBayes()
    #     classifier.train(train)
    #     classifier.save("./models/imbl_multinomial_laplace_no_stop_words")
    #     evaluate(classifier, test)




if __name__ == "__main__":
    main(verbosity=2)