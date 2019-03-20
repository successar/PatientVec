from PatientVec.preprocess.vectorizer import BoWder
from sklearn.linear_model import LogisticRegression

from sklearn.multioutput import MultiOutputClassifier
from PatientVec.metrics import *
import time, json
import os

def normalise_output(y) :
    if y.shape[0] == 1 :
        return y[0]
    return y[:, :, 1].T

class Classifier :
    def __init__(self, lr_model, name, dirname) :
        self.name = name
        self.dirname = dirname
        self.classifier = MultiOutputClassifier(LogisticRegression(class_weight='balanced', penalty='l1'), n_jobs=8)
        self.metrics = lr_model.metrics

    def evaluate(self, X, y, save_results) :
        pred = normalise_output(np.array(self.classifier.predict_proba(X)))
        metrics = self.metrics(y, pred)
        print(self.name)
        print_metrics(metrics)
        if save_results :
            os.makedirs(self.dirname, exist_ok=True)
            f = open(self.dirname + '/evaluate.json', 'w')
            json.dump(metrics, f)
            f.close()
            
        self.test_metrics = metrics
    

class LR :
    def __init__(self, config) :
        vocab = config['vocab']
        stop_words = config.get('stop_words', False)
        self.metrics = metrics_map[config['type']]
        self.norm = config.get('norm', None)
        self.constant_mul = config.get('constant_mul', 1.0)
        self.has_structured = config.get('structured', True)

        self.time_str = time.ctime().replace(' ', '_')
        self.exp_name = config['exp_name']

        self.bowder = BoWder(vocab=vocab, stop_words=stop_words, norm=self.norm, constant_mul=self.constant_mul)

        gen_dirname = lambda x : os.path.join('outputs/', self.exp_name, 'baselines', x, self.time_str)
        bow_dirname = gen_dirname('LR+BOW+norm='+str(self.norm))
        tf_dirname = gen_dirname('LR+TFIDF+norm='+str(self.norm))
        binbow_dirname = gen_dirname('LR+BinaryBOW+norm='+str(self.norm))
        bow_structured_dirname = gen_dirname('LR+BOW+norm=' + str(self.norm) + '+Structured')
        binbow_structured_dirname = gen_dirname('LR+BinaryBOW+norm=' + str(self.norm) + '+Structured')
        tf_structured_dirname = gen_dirname('LR+TFIDF+norm=' + str(self.norm) + '+Structured')
        structured_dirname = gen_dirname('LR+Structured')
        
        self.bow_classifier = Classifier(self, 'BOW', bow_dirname)
        self.tf_idf_classifier = Classifier(self, 'TFIDF', tf_dirname)
        self.binbow_classifier = Classifier(self, 'BinBOW', binbow_dirname)
        self.bow_with_structured_classifier = Classifier(self, 'BOW+Structured', bow_structured_dirname)
        self.binbow_with_structured_classifier = Classifier(self, 'BinBOW+Structured', binbow_structured_dirname)
        self.tf_idf_with_structured_classifier = Classifier(self, 'TFIDF+Structured', tf_structured_dirname)
        self.structured_classifier = Classifier(self, 'Structured', structured_dirname)

    def train(self, train_data) :
        docs = [[y for x in d for y in x] for d in train_data.X]
        
        train_bow = self.bowder.get_bow(docs)
        self.bow_classifier.classifier.fit(train_bow, train_data.y)

        if self.has_structured :
            train_bow = np.concatenate([np.array(train_bow), train_data.structured_data], axis=-1)
            self.bow_with_structured_classifier.classifier.fit(train_bow, train_data.y)

        del train_bow

        train_bow = self.bowder.get_binary_bow(docs)
        self.binbow_classifier.classifier.fit(train_bow, train_data.y)
        
        if self.has_structured :
            train_bow = np.concatenate([np.array(train_bow), train_data.structured_data], axis=-1)
            self.binbow_with_structured_classifier.classifier.fit(train_bow, train_data.y)

        del train_bow

        self.bowder.fit_tfidf(docs)
        train_tf = self.bowder.get_tfidf(docs)

        self.tf_idf_classifier.classifier.fit(train_tf, train_data.y)
        if self.has_structured :
            train_tf = np.concatenate([np.array(train_tf), train_data.structured_data], axis=-1)
            self.tf_idf_with_structured_classifier.classifier.fit(train_tf, train_data.y)

        del train_tf

        if self.has_structured :
            self.structured_classifier.classifier.fit(train_data.structured_data, train_data.y)

    def evaluate(self, data, save_results=False) :
        docs = [[y for x in d for y in x] for d in data.X]

        train_bow = self.bowder.get_bow(docs)
        self.bow_classifier.evaluate(train_bow, data.y, save_results)

        if self.has_structured :
            train_bow = np.concatenate([np.array(train_bow), data.structured_data], axis=-1)
            self.bow_with_structured_classifier.evaluate(train_bow, data.y, save_results)

        del train_bow

        train_bow = self.bowder.get_binary_bow(docs)
        self.binbow_classifier.evaluate(train_bow, data.y, save_results)
        
        if self.has_structured :
            train_bow = np.concatenate([np.array(train_bow), data.structured_data], axis=-1)
            self.binbow_with_structured_classifier.evaluate(train_bow, data.y, save_results)

        del train_bow

        self.bowder.fit_tfidf(docs)
        train_tf = self.bowder.get_tfidf(docs)

        self.tf_idf_classifier.evaluate(train_tf, data.y, save_results)
        if self.has_structured :
            train_tf = np.concatenate([np.array(train_tf), data.structured_data], axis=-1)
            self.tf_idf_with_structured_classifier.evaluate(train_tf, data.y, save_results)

        del train_tf

        if self.has_structured :
            self.structured_classifier.evaluate(data.structured_data, data.y, save_results)
            
    def predict(self, data) :
        docs = [[y for x in d for y in x] for d in data.X]
        bow = self.bowder.get_bow(docs)
        pred = normalise_output(np.array(self.bow_classifier.classifier.predict_proba(bow)))
        return pred

    def get_features(self, classifier, estimator=0, n=100) :
        return [self.bowder.vocab.idx2word[self.bowder.map_bow_to_vocab[x]] for x in 
                    np.argsort(classifier.classifier.estimators_[estimator].coef_[0][:len(self.bowder.words_to_keep)])[-n:]]

    def print_all_features(self, n=20) :
        for i in range(len(self.bow_classifier.classifier.estimators_)) :
            print(" ".join(self.get_features(self.bow_classifier, estimator=i, n=n)))
            print('-'*10)
            print(" ".join(self.get_features(self.tf_idf_classifier, estimator=i, n=n)))
            print('-'*10)
            print(" ".join(self.get_features(self.bow_with_structured_classifier, estimator=i, n=n)))
            print('-'*10)
            print(" ".join(self.get_features(self.tf_idf_with_structured_classifier, estimator=i, n=n)))
            print('-'*10)

            print('='*25)


from sklearn.decomposition import LatentDirichletAllocation

class LDA :
    def __init__(self, config) :
        vocab = config['vocab']
        stop_words = config.get('stop_words', False)
        self.metrics = metrics_map[config['type']]

        self.time_str = time.ctime().replace(' ', '_')
        self.exp_name = config['exp_name']

        self.bowder = BoWder(vocab=vocab, stop_words=stop_words)
        self.lda = LatentDirichletAllocation(n_components=50, learning_method='online', verbose=1)

        self.lda_classifier = MultiOutputClassifier(LogisticRegression(class_weight='balanced', penalty='l1'), n_jobs=4)
        self.lda_with_structured_classifier = MultiOutputClassifier(LogisticRegression(class_weight='balanced', penalty='l1'), n_jobs=4)
        
        self.dirname = os.path.join('outputs/', self.exp_name, 'baselines', 'LR+LDA', self.time_str)
        self.structured_dirname = os.path.join('outputs/', self.exp_name, 'baselines', 'LR+LDA+Structured', self.time_str)

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

        pred_lda = normalise_output(np.array(self.lda_classifier.predict_proba(dev_lda)))

        dev_lda = np.concatenate([dev_lda, data.structured_data], axis=-1)
        pred_lda_structured = normalise_output(np.array(self.lda_with_structured_classifier.predict_proba(dev_lda)))

        metrics = self.metrics(data.y, pred_lda)
        metrics_structured = self.metrics(data.y, pred_lda_structured)
        print("LDA")
        print_metrics(metrics)
        print("LDA_Structured")
        print_metrics(metrics_structured)

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


