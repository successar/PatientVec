from PatientVec.preprocess.vectorizer import BoWder
from sklearn.linear_model import LogisticRegression
from PatientVec.metrics import *
import time, json
import os

class LR :
    def __init__(self, config) :
        vocab = config['vocab']
        stop_words = config.get('stop_words', False)
        encodings = config['encodings']

        self.bowder = BoWder(vocab=vocab, stop_words=stop_words)
        self.bow_classifier = LogisticRegression(class_weight='balanced', penalty='l1')
        self.tf_idf_classifier = LogisticRegression(class_weight='balanced', penalty='l1')

        self.time_str = time.ctime().replace(' ', '_')
        self.exp_name = config['exp_name']
        self.bow_dirname = 'outputs/LR/BOW/' + ".".join(encodings) + '/' + self.exp_name + '/' + self.time_str
        self.tf_dirname = 'outputs/LR/TF/' + ".".join(encodings) + '/' + self.exp_name + '/' + self.time_str

    def train(self, train_data) :
        self.bowder.fit_tfidf(train_data.X)
        train_bow = self.bowder.get_bow(train_data.X)
        train_tf = self.bowder.get_tfidf(train_data.X)

        train_cond = train_data.cond
        train_bow = np.concatenate([train_bow.todense(), train_cond], axis=-1)
        train_tf = np.concatenate([train_tf.todense(), train_cond], axis=-1)

        self.bow_classifier.fit(train_bow, train_data.y)
        self.tf_idf_classifier.fit(train_tf, train_data.y)

        self.n_encodings = train_cond.shape[-1]

    def evaluate(self, data, save_results=False) :
        bow = self.bowder.get_bow(data.X)
        tf = self.bowder.get_tfidf(data.X)

        cond = data.cond
        bow = np.concatenate([bow.todense(), cond], axis=-1)
        tf = np.concatenate([tf.todense(), cond], axis=-1)

        pred_bow = self.bow_classifier.predict_proba(bow)
        pred_tf = self.tf_idf_classifier.predict_proba(tf)

        metric_bow = calc_metrics_classification(data.y, pred_bow)
        metric_tf = calc_metrics_classification(data.y, pred_tf)

        print(metric_bow)
        print(metric_tf)

        if save_results :
            os.makedirs(self.bow_dirname, exist_ok=True)
            f = open(self.bow_dirname + '/evaluate.json', 'w')
            json.dump(metric_bow, f)
            f.close()

            os.makedirs(self.tf_dirname, exist_ok=True)
            f = open(self.tf_dirname + '/evaluate.json', 'w')
            json.dump(metric_tf, f)
            f.close()

    def get_features(self, n=100) :
        vsize = self.bowder.vocab.vocab_size
        return [self.bowder.vocab.idx2word[self.bowder.map_bow_to_vocab[x]] for x in np.argsort(self.tf_idf_classifier.coef_[0][:vsize])[-n:]]

from sklearn.decomposition import LatentDirichletAllocation

class LDA :
    def __init__(self, config) :
        vocab = config['vocab']
        stop_words = config.get('stop_words', False)
        encodings = config['encodings']

        self.bowder = BoWder(vocab=vocab, stop_words=stop_words)
        self.lda = LatentDirichletAllocation(n_components=50, learning_method='online', verbose=1)
        self.lda_classifier = LogisticRegression(class_weight='balanced', penalty='l1')

        self.time_str = time.ctime().replace(' ', '_')
        self.exp_name = config['exp_name']
        self.dirname = 'outputs/LDA/' + ".".join(encodings) + '/' + self.exp_name + '/' + self.time_str

    def train(self, train_data) :
        train_bow = self.bowder.get_bow(train_data.X)
        self.lda.fit(train_bow)

        train_lda = self.lda.transform(train_bow)

        train_cond = train_data.cond
        train_lda = np.concatenate([train_lda, train_cond], axis=-1)

        self.lda_classifier.fit(train_lda, train_data.y)

    def evaluate(self, data, save_results=False) :
        dev_bow = self.bowder.get_bow(data.X)
        dev_lda = self.lda.transform(dev_bow)

        cond = data.cond
        dev_lda = np.concatenate([dev_lda, cond], axis=-1)

        pred_lda = self.lda_classifier.predict_proba(dev_lda)

        metrics = calc_metrics_classification(data.y, pred_lda)
        print(metrics)
        if save_results :
            os.makedirs(self.dirname, exist_ok=True)
            f = open(self.dirname + '/evaluate.json', 'w')
            json.dump(metrics, f)
            f.close()

    def get_topics(self, n=20) :
        importance_scores = self.lda.components_ / self.lda.components_.sum(axis=1)[:, np.newaxis]
        topic_best = {}
        for i in range(50) :
            topic_dist = importance_scores[i]
            topic_best[i] = " ".join([self.bowder.vocab.idx2word[self.bowder.map_bow_to_vocab[x]] for x in np.argsort(topic_dist)[-n:]])

        
        return topic_best




