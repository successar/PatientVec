from PatientVec.preprocess.vectorizer import BoWder
from sklearn.linear_model import LogisticRegression
from PatientVec.metrics import *
import time, json
import os

class LR :
    def __init__(self, config) :
        vocab = config['vocab']
        stop_words = config.get('stop_words', False)

        self.bowder = BoWder(vocab=vocab, stop_words=stop_words)
        self.bow_classifier = LogisticRegression(class_weight='balanced', penalty='l1')
        self.tf_idf_classifier = LogisticRegression(class_weight='balanced', penalty='l1')

        self.bow_with_structured_classifier = LogisticRegression(class_weight='balanced', penalty='l1')
        self.tf_idf_with_structured_classifier = LogisticRegression(class_weight='balanced', penalty='l1')

        self.time_str = time.ctime().replace(' ', '_')
        self.exp_name = config['exp_name']

        self.bow_dirname = os.path.join('outputs/baselines/', self.exp_name, 'baselines', 'LR+BOW', self.time_str)
        self.tf_dirname = os.path.join('outputs/baselines/', self.exp_name, 'baselines', 'LR+TFIDF', self.time_str)

        self.bow_structured_dirname = os.path.join('outputs/baselines/', self.exp_name, 'baselines', 'LR+BOW+Structured', self.time_str)
        self.tf_structured_dirname = os.path.join('outputs/baselines/', self.exp_name, 'baselines', 'LR+TFIDF+Structured', self.time_str)

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

    def get_features(self, n=100) :
        return [self.bowder.vocab.idx2word[self.bowder.map_bow_to_vocab[x]] for x in np.argsort(self.tf_idf_classifier.coef_[0])[-n:]]

from sklearn.decomposition import LatentDirichletAllocation

class LDA :
    def __init__(self, config) :
        vocab = config['vocab']
        stop_words = config.get('stop_words', False)

        self.bowder = BoWder(vocab=vocab, stop_words=stop_words)
        self.lda = LatentDirichletAllocation(n_components=50, learning_method='online', verbose=1)

        self.lda_classifier = LogisticRegression(class_weight='balanced', penalty='l1')
        self.lda_with_structured_classifier = LogisticRegression(class_weight='balanced', penalty='l1')

        self.time_str = time.ctime().replace(' ', '_')
        self.exp_name = config['exp_name']
        self.dirname = os.path.join('outputs/baselines/', self.exp_name, 'baselines', 'LR+LDA', self.time_str)
        self.structured_dirname = os.path.join('outputs/baselines/', self.exp_name, 'baselines', 'LR+LDA+Structured', self.time_str)

    def train(self, train_data) :
        docs = [[y for x in d for y in x] for d in train_data.X]
        train_bow = self.bowder.get_bow(docs)
        self.lda.fit(train_bow)

        train_lda = self.lda.transform(train_bow)
        self.lda_classifier.fit(train_lda, train_data.y)

        train_lda = np.concatenate([train_lda, train_data.structured_data], axis=-1)
        self.lda_with_structured_classifier.fit(train_lda, train_data.y)

    def evaluate(self, data, save_results=False) :
        docs = [[y for x in d for y in x] for d in data.X]
        dev_bow = self.bowder.get_bow(docs)
        dev_lda = self.lda.transform(dev_bow)

        pred_lda = self.lda_classifier.predict_proba(dev_lda)

        dev_lda = np.concatenate([dev_lda, data.structured_data], axis=-1)
        pred_lda_structured = self.lda_with_structured_classifier.predict_proba(dev_lda)

        metrics = calc_metrics_classification(data.y, pred_lda)
        metrics_structured = calc_metrics_classification(data.y, pred_lda_structured)
        print("LDA", metrics)
        print("LDA_Structured", metrics_structured)

        if save_results :
            os.makedirs(self.dirname, exist_ok=True)
            f = open(self.dirname + '/evaluate.json', 'w')
            json.dump(metrics, f)
            f.close()

            os.makedirs(self.structured_dirname, exist_ok=True)
            f = open(self.structured_dirname + '/evaluate.json', 'w')
            json.dump(metrics_structured, f)
            f.close()

    def get_topics(self, n=20) :
        importance_scores = self.lda.components_ / self.lda.components_.sum(axis=1)[:, np.newaxis]
        topic_best = {}
        for i in range(50) :
            topic_dist = importance_scores[i]
            topic_best[i] = " ".join([self.bowder.vocab.idx2word[self.bowder.map_bow_to_vocab[x]] for x in np.argsort(topic_dist)[-n:]])

        
        return topic_best


