from PatientVec.preprocess.vectorizer import BoWder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from PatientVec.metrics import *
import time, json
import os

class MLP :
    def __init__(self, config) :
        vocab = config['vocab']
        stop_words = config.get('stop_words', False)

        self.bowder = BoWder(vocab=vocab, stop_words=stop_words)
        self.bow_classifier = MLPClassifier(class_weight='balanced', penalty='l1')
        self.tf_idf_classifier = MLPClassifier(class_weight='balanced', penalty='l1')

        self.bow_with_structured_classifier = MLPClassifier(class_weight='balanced', penalty='l1')
        self.tf_idf_with_structured_classifier = MLPClassifier(class_weight='balanced', penalty='l1')

        self.time_str = time.ctime().replace(' ', '_')
        self.exp_name = config['exp_name']

        self.bow_dirname = os.path.join('outputs/baselines/', self.exp_name, 'baselines', 'MLP+BOW', self.time_str)
        self.tf_dirname = os.path.join('outputs/baselines/', self.exp_name, 'baselines', 'MLP+TFIDF', self.time_str)

        self.bow_structured_dirname = os.path.join('outputs/baselines/', self.exp_name, 'baselines', 'MLP+BOW+Structured', self.time_str)
        self.tf_structured_dirname = os.path.join('outputs/baselines/', self.exp_name, 'baselines', 'MLP+TFIDF+Structured', self.time_str)

    def train(self, train_data) :
        docs = [[y for x in d for y in x] for d in train_data.X]
        self.bowder.fit_tfidf(docs)
        train_bow = self.bowder.get_bow(docs)
        train_tf = self.bowder.get_tfidf(docs)

        self.bow_classifier.fit(train_bow, train_data.y)
        self.tf_idf_classifier.fit(train_tf, train_data.y)

        train_bow = np.concatenate([train_bow.todense(), train_data.structured_data], axis=-1)
        train_tf = np.concatenate([train_tf.todense(), train_data.structured_data], axis=-1)

        self.bow_with_structured_classifier.fit(train_bow, train_data.y)
        self.tf_idf_with_structured_classifier.fit(train_tf, train_data.y)

    def evaluate(self, data, save_results=False) :
        docs = [[y for x in d for y in x] for d in data.X]
        bow = self.bowder.get_bow(docs)
        tf = self.bowder.get_tfidf(docs)

        pred_bow = self.bow_classifier.predict_proba(bow)
        pred_tf = self.tf_idf_classifier.predict_proba(tf)

        bow = np.concatenate([bow.todense(), data.structured_data], axis=-1)
        tf = np.concatenate([tf.todense(), data.structured_data], axis=-1)

        pred_bow_structured = self.bow_with_structured_classifier.predict_proba(bow)
        pred_tf_structured = self.tf_idf_with_structured_classifier.predict_proba(tf)

        metric_bow = calc_metrics_classification(data.y, pred_bow)
        metric_tf = calc_metrics_classification(data.y, pred_tf)

        metric_bow_structured = calc_metrics_classification(data.y, pred_bow_structured)
        metric_tf_structured = calc_metrics_classification(data.y, pred_tf_structured)

        print("Bow", metric_bow)
        print("TFIDF", metric_tf)

        print("Bow_structured", metric_bow_structured)
        print("TFIDF_structured", metric_tf_structured)

        if save_results :
            os.makedirs(self.bow_dirname, exist_ok=True)
            f = open(self.bow_dirname + '/evaluate.json', 'w')
            json.dump(metric_bow, f)
            f.close()

            os.makedirs(self.tf_dirname, exist_ok=True)
            f = open(self.tf_dirname + '/evaluate.json', 'w')
            json.dump(metric_tf, f)
            f.close()

            os.makedirs(self.bow_structured_dirname, exist_ok=True)
            f = open(self.bow_structured_dirname + '/evaluate.json', 'w')
            json.dump(metric_bow_structured, f)
            f.close()

            os.makedirs(self.tf_structured_dirname, exist_ok=True)
            f = open(self.tf_structured_dirname + '/evaluate.json', 'w')
            json.dump(metric_tf_structured, f)
            f.close()