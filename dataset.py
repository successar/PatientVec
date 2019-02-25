import pandas as pd 
import pickle
import re
import numpy as np
import os
from PatientVec.preprocess.vocabulary import Vocabulary
from PatientVec.preprocess.embedder import PretrainedEmbedding
from PatientVec.preprocess.vectorizer import DataHolder, BoWder
from PatientVec.preprocess.field_processor import field_processors

import logging
logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)

class Dataset() :
    def __init__(self, name, dirname) :
        self.name = name
        self.vocab = Vocabulary().load(dirname)
        self.embedding = PretrainedEmbedding().load(dirname)

        logging.info("Reading Structured data ...")
        structured_data_file = os.path.join(dirname, 'Split_Structured_Final.msg')
        self.dataframe = pd.read_msgpack(structured_data_file).reset_index(drop=True)
        assert 'exp_split' in self.dataframe.columns, logging.error("exp_split not in dataframe columns")

        logging.info("Reading Notes ...")
        notes_file = os.path.join(dirname, 'combined_notes_sequences.p')
        self.sequences = pickle.load(open(notes_file, 'rb'))

        logging.info("Stratifying ...")
        self.idxs = {}
        for t in ['train', 'dev', 'test'] :
            self.idxs[t] = list(self.dataframe[self.dataframe['exp_split'] == t].index)
            if len(self.idxs[t]) == 0 : 
                logging.warning("No records for %s split", t)

        self.encodings = {}
        self.structured_columns = []
        self.structured_dim = 0
        
    def generate_labels(self, label_list, output_size, predictor) :
        for labelfield in label_list :
            assert labelfield in self.dataframe.columns, logging.error("Labelfield %s not in dataframe columns", labelfield)
        label_cols = np.array([list(self.dataframe[l]) for l in label_list]).T
        self.y_header = label_list
        self.y = label_cols
        
        self.output_size = output_size
        self.predictor_type = predictor

    def generate_bowder(self, data, stop_words=True, norm=None) :
        self.bowder = BoWder(vocab=self.vocab, stop_words=stop_words, norm=norm)
        docs = [[y for x in d for y in x] for d in data.X]
        self.bowder.fit_tfidf(docs)

    def get_vec_encoding(self, data, _type='bow') :
        docs = [[y for x in d for y in x] for d in data.X]
        if _type == 'bow' :
            return self.bowder.get_bow(docs)
        elif _type == 'tfidf' :
            return self.bowder.get_tfidf(docs)
        else :
            raise LookupError("No such encoding")

    def generate_encoded_field(self, field, encoding_type, encoding_args=None) :
        assert field in self.dataframe.columns, logging.error("%s not in dataframe columns", field)
        if encoding_args is None : 
            encoding_args = {}

        datafield = np.array(self.dataframe[field])
        encoder = field_processors[encoding_type](encoding_args)
        encoder.fit(datafield)

        self.encodings[field] = encoder

    def set_structured_params(self, regexs) :
        columns = list(self.dataframe.columns)
        self.structured_columns = [x for x in columns if any(re.search(r, x) for r in regexs)]
        self.structured_dim = 0
        for x in self.structured_columns :
            assert x in self.encodings, logging.error('%s not in encodings', x)
            self.structured_dim += self.encodings[x].get_output_dim()

    def get_embedding_params(self) :
        return {
            "vocab_size" : self.vocab.vocab_size,
            "embed_size" : self.embedding.word_dim,
            "embedding_file" : self.embedding.embedding_file
        }

    def get_encodings_dim(self, encodings) :
        encoding_dim = 0
        for e in encodings :
            assert e in self.encodings, logging.error('%s field not in encodings', e)
            encoding_dim += self.encodings[e].get_output_dim()

        return encoding_dim

    def get_data(self, _type, structured, encodings=None) :
        filtered_idxs = self.idxs[_type]
        X = [self.sequences[i] for i in filtered_idxs]
        y = [self.y[i] for i in filtered_idxs]

        print("Pos Percentage", np.array(sum(y))/len(y))

        data = DataHolder(X=X, y=y)
        
        if encodings is not None :
            encoded_values = []
            for k in encodings :
                field = np.array(self.dataframe[k])
                filtered_field = np.array([field[i] for i in filtered_idxs])
                encoded_values.append(self.encodings[k].transform(filtered_field))

            encoded_values = np.concatenate(encoded_values, axis=1)
            data.add_fields(cond=encoded_values)

        if structured :
            structured_values = []
            for k in self.structured_columns :
                field = np.array(self.dataframe[k])
                filtered_field = np.array([field[i] for i in filtered_idxs])
                structured_values.append(self.encodings[k].transform(filtered_field))

            structured_values = np.concatenate(structured_values, axis=1)
            data.add_fields(structured_data=structured_values)

        return data

    def filter_data_length(self, data, truncate=95) :
        docs = [[y for x in d for y in x] for d in data.X]
        total_sentence_length = [len(x) for x in docs]
        
        filter_perc = np.percentile(total_sentence_length, truncate)
        logging.info("Maximum Sentence Length %f , %d percentile length %f ... ", max(total_sentence_length), truncate, filter_perc)
        
        data.X = [[d[-int(filter_perc):]] for d in docs]
        return data