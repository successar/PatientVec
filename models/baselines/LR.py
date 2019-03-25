from PatientVec.preprocess.vectorizer import BoWder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.multioutput import MultiOutputClassifier
from PatientVec.metrics import *
import time, json, pickle
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

    def fit(self, X, y) :
        print("Fitting ... ", self.name)
        self.classifier.fit(X, y)

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
        self.has_structured = config.get('has_structured', True)
        self.only_structured = config.get('only_structured', False)
        self.basepath = config.get('basepath', 'outputs')

        self.time_str = time.ctime().replace(' ', '_')
        self.exp_name = config['exp_name']
        
        self.gen_dirname = lambda x : os.path.join(self.basepath, self.exp_name, 'baselines', x, self.time_str)
        if config.get('lda', False) :
            print("Setting up LDA ...")
            self.init_lda(config)
            return
        
        self.methods = config.get('methods', ['count', 'binary', 'tfidf'])
        print("Not running LDA, Yay !")
        self.bowder = BoWder(vocab=vocab, stop_words=stop_words, norm=self.norm, constant_mul=self.constant_mul)
        bow_dirname = self.gen_dirname('LR+BOW+norm='+str(self.norm))
        tf_dirname = self.gen_dirname('LR+TFIDF+norm='+str(self.norm))
        binbow_dirname = self.gen_dirname('LR+BinaryBOW+norm='+str(self.norm))
        bow_structured_dirname = self.gen_dirname('LR+BOW+norm=' + str(self.norm) + '+Structured')
        binbow_structured_dirname = self.gen_dirname('LR+BinaryBOW+norm=' + str(self.norm) + '+Structured')
        tf_structured_dirname = self.gen_dirname('LR+TFIDF+norm=' + str(self.norm) + '+Structured')
        structured_dirname = self.gen_dirname('LR+Structured')
        
        self.bow_classifier = Classifier(self, 'BOW', bow_dirname)
        self.tf_idf_classifier = Classifier(self, 'TFIDF', tf_dirname)
        self.binbow_classifier = Classifier(self, 'BinBOW', binbow_dirname)
        self.bow_with_structured_classifier = Classifier(self, 'BOW+Structured', bow_structured_dirname)
        self.binbow_with_structured_classifier = Classifier(self, 'BinBOW+Structured', binbow_structured_dirname)
        self.tf_idf_with_structured_classifier = Classifier(self, 'TFIDF+Structured', tf_structured_dirname)
        self.structured_classifier = Classifier(self, 'Structured', structured_dirname)

    def init_lda(self, config) :
        vocab = config['vocab']
        stop_words = config.get('stop_words', False)
        self.bowder = BoWder(vocab=vocab, stop_words=stop_words, norm=None, constant_mul=self.constant_mul)
        lda_dirname = self.gen_dirname('LR+LDA+norm=None')
        lda_l2_dirname = self.gen_dirname('LR+LDA+norm=l2')
        lda_structured_dirname = self.gen_dirname('LR+LDA+norm=None+Structured')
        lda_l2_structured_dirname = self.gen_dirname('LR+LDA+norm=l2+Structured')

        self.lda_classifier = Classifier(self, 'LDA', lda_dirname)
        self.lda_l2_classifier = Classifier(self, 'LDA+l2', lda_l2_dirname)
        self.lda_structured_classifier = Classifier(self, 'LDA+Structured', lda_structured_dirname)
        self.lda_l2_structured_classifier = Classifier(self, 'LDA+l2+Structured', lda_l2_structured_dirname)

        self.lda_file = os.path.join(os.path.dirname(lda_dirname), 'lda_object.p')
        os.makedirs(os.path.dirname(self.lda_file), exist_ok=True)
        print(self.lda_file)

    def train_lda(self, train_data) :
        docs = [[y for x in d for y in x] for d in train_data.X]
        
        train_bow = self.bowder.get_bow(docs)
        self.lda = LatentDirichletAllocation(n_components=50, learning_method='online', verbose=1)
        self.lda.fit(train_bow)
        pickle.dump(self.lda, open(self.lda_file, 'wb'))

        train_lda = self.lda.transform(train_bow)

        self.lda_classifier.fit(train_lda, train_data.y)
        if self.has_structured :
            train_lda = np.concatenate([np.array(train_lda), train_data.structured_data], axis=-1)
            self.lda_structured_classifier.fit(train_lda, train_data.y)

        train_lda = self.lda.transform(train_bow)
        train_lda = self.bowder.normalise_bow(train_lda, use_norm='l2')

        self.lda_l2_classifier.fit(train_lda, train_data.y)
        if self.has_structured :
            train_lda = np.concatenate([np.array(train_lda), train_data.structured_data], axis=-1)
            self.lda_l2_structured_classifier.fit(train_lda, train_data.y)

    def evaluate_lda(self, data, save_results=False) :
        docs = [[y for x in d for y in x] for d in data.X]

        train_bow = self.bowder.get_bow(docs)
        train_lda = self.lda.transform(train_bow)

        self.lda_classifier.evaluate(train_lda, data.y, save_results)
        if self.has_structured :
            train_lda = np.concatenate([np.array(train_lda), data.structured_data], axis=-1)
            self.lda_structured_classifier.evaluate(train_lda, data.y, save_results)

        train_lda = self.lda.transform(train_bow)
        train_lda = self.bowder.normalise_bow(train_lda, use_norm='l2')

        self.lda_l2_classifier.evaluate(train_lda, data.y, save_results)
        if self.has_structured :
            train_lda = np.concatenate([np.array(train_lda), data.structured_data], axis=-1)
            self.lda_l2_structured_classifier.evaluate(train_lda, data.y, save_results)

    def train(self, train_data) :
        docs = [[y for x in d for y in x] for d in train_data.X]
        
        if 'count' in self.methods :
            train_bow = self.bowder.get_bow(docs)
            if not self.only_structured :
                self.bow_classifier.fit(train_bow, train_data.y)

            if self.has_structured :
                train_bow = np.concatenate([np.array(train_bow), train_data.structured_data], axis=-1)
                self.bow_with_structured_classifier.fit(train_bow, train_data.y)

            del train_bow

        if 'binary' in self.methods :
            train_bow = self.bowder.get_binary_bow(docs)
            if not self.only_structured :
                self.binbow_classifier.fit(train_bow, train_data.y)

            if self.has_structured :
                train_bow = np.concatenate([np.array(train_bow), train_data.structured_data], axis=-1)
                self.binbow_with_structured_classifier.fit(train_bow, train_data.y)

            del train_bow
            
        if 'tfidf' in self.methods :
            self.bowder.fit_tfidf(docs)
            train_tf = self.bowder.get_tfidf(docs)

            if not self.only_structured :
                self.tf_idf_classifier.fit(train_tf, train_data.y)
                
            if self.has_structured :
                train_tf = np.concatenate([np.array(train_tf), train_data.structured_data], axis=-1)
                self.tf_idf_with_structured_classifier.fit(train_tf, train_data.y)

            del train_tf

        if self.has_structured :
            self.structured_classifier.fit(train_data.structured_data, train_data.y)

    def evaluate(self, data, save_results=False) :
        docs = [[y for x in d for y in x] for d in data.X]

        if 'count' in self.methods :
            train_bow = self.bowder.get_bow(docs)
            if not self.only_structured :
                self.bow_classifier.evaluate(train_bow, data.y, save_results)

            if self.has_structured :
                train_bow = np.concatenate([np.array(train_bow), data.structured_data], axis=-1)
                self.bow_with_structured_classifier.evaluate(train_bow, data.y, save_results)

            del train_bow

        if 'binary' in self.methods :
            train_bow = self.bowder.get_binary_bow(docs)
            if not self.only_structured :
                self.binbow_classifier.evaluate(train_bow, data.y, save_results)

            if self.has_structured :
                train_bow = np.concatenate([np.array(train_bow), data.structured_data], axis=-1)
                self.binbow_with_structured_classifier.evaluate(train_bow, data.y, save_results)

            del train_bow

        if 'tfidf' in self.methods :
            self.bowder.fit_tfidf(docs)
            train_tf = self.bowder.get_tfidf(docs)

            if not self.only_structured :
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
