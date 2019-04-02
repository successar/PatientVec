from sklearn.linear_model import LogisticRegression
from PatientVec.metrics import *
import time, json
import os

class LR_Structured :
    def __init__(self, config) :
        self.classifier = LogisticRegression(class_weight='balanced', penalty='l2')
        self.time_str = time.ctime().replace(' ', '_')
        self.exp_name = config['exp_name']
        self.dirname = 'outputs/LR/Structured/' + self.exp_name + '/' + self.time_str

    def train(self, train_data) :
        structured_data = np.array(train_data.structured_data)
        self.classifier.fit(structured_data, train_data.y)

    def evaluate(self, data, save_results=False) :
        structured_data = np.array(data.structured_data)
        pred = self.classifier.predict_proba(structured_data)
        metric = calc_metrics_classification(data.y, pred)
        print(metric)

        if save_results :
            os.makedirs(self.dirname, exist_ok=True)
            f = open(self.dirname + '/evaluate.json', 'w')
            json.dump(metric, f)
            f.close()

    def get_top_features(self) :
        return [x for x in np.argsort(self.classifier.coef_[0])]