from .utils import get_sorting_index_with_noise_from_lengths
import random
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class Batch() :
    def __init__(self, **kwargs) :
        for n, v in kwargs.items() :
            setattr(self, n, v)

    def have(self, attr) :
        return hasattr(self, attr)

class Vector_Generator() :
    def __init__(self, train_data, batch_size, sort_and_shuffle=True) :
        docs = train_data.X

        self.valid_idxs = list(range(len(docs)))
        self.N = len(self.valid_idxs)
        self.batch_size = batch_size

        self.batches = [self.valid_idxs[i:i+batch_size] for i in range(0, len(self.valid_idxs), batch_size)]

        if sort_and_shuffle :
            random.shuffle(self.batches)

        self.batch_num = 0
        self.train_data = train_data
        
    def __iter__(self) :
        return self

    def __next__(self):
        if self.batch_num >= len(self.batches) :
            raise StopIteration
        else :
            batch = self.generate_batch(self.batch_num)
            self.batch_num += 1
            return batch

    def __len__(self) :
        return len(self.batches)

    def generate_batch(self, batch_num) :
        batch_fields = {}
        idxs = self.batches[batch_num]
        for att in self.train_data.attributes :
            att_value = getattr(self.train_data, att)
            batch_fields[att] = [att_value[i] for i in idxs]
        
        return Batch(**batch_fields)

class Hierarchical_Generator() :
    def __init__(self, train_data, batch_size, sort_and_shuffle=True) :
        docs = train_data.X

        self.valid_idxs = list(range(len(docs)))

        if sort_and_shuffle :
            max_sentence_length = [max([len(x) for x in y]) for y in docs]
            sorting_idx = get_sorting_index_with_noise_from_lengths(max_sentence_length, noise_frac=0.1)
            self.valid_idxs = [self.valid_idxs[i] for i in sorting_idx]

        self.N = len(self.valid_idxs)
        self.batch_size = batch_size

        self.batches = [self.valid_idxs[i:i+batch_size] for i in range(0, len(self.valid_idxs), batch_size)]

        if sort_and_shuffle :
            random.shuffle(self.batches)

        self.batch_num = 0
        
        self.train_data = train_data
        
    def __iter__(self) :
        return self

    def __next__(self):
        if self.batch_num >= len(self.batches) :
            raise StopIteration
        else :
            batch = self.generate_batch(self.batch_num)
            self.batch_num += 1
            return batch

    def __len__(self) :
        return len(self.batches)

    def generate_batch(self, batch_num) :
        batch_fields = {}
        idxs = self.batches[batch_num]
        for att in self.train_data.attributes :
            att_value = getattr(self.train_data, att)
            batch_fields[att] = [att_value[i] for i in idxs]
        
        return Batch(**batch_fields)

class Concatenated_Generator() :
    def __init__(self, train_data, batch_size, sort_and_shuffle=True) :
        docs = [[y for x in d for y in x] for d in train_data.X]

        self.valid_idxs = list(range(len(docs)))

        if sort_and_shuffle :
            valid_lengths = [len(docs[i]) for i in self.valid_idxs]
            sorting_idx = get_sorting_index_with_noise_from_lengths(valid_lengths, noise_frac=0.1)
            self.valid_idxs = [self.valid_idxs[i] for i in sorting_idx]

        self.N = len(self.valid_idxs)
        self.batch_size = batch_size

        self.batches = [self.valid_idxs[i:i+batch_size] for i in range(0, len(self.valid_idxs), batch_size)]

        if sort_and_shuffle :
            random.shuffle(self.batches)

        self.batch_num = 0
        
        self.train_data = train_data

    def __iter__(self) :
        return self

    def __next__(self):
        if self.batch_num >= len(self.batches) :
            raise StopIteration
        else :
            batch = self.generate_batch(self.batch_num)
            self.batch_num += 1
            return batch

    def __len__(self) :
        return len(self.batches)

    def generate_batch(self, batch_num) :
        batch_fields = {}
        idxs = self.batches[batch_num]
        for att in self.train_data.attributes :
            att_value = getattr(self.train_data, att)
            batch_fields[att] = [att_value[i] for i in idxs]

        batch_fields['X'] = [[y for x in d for y in x] for d in batch_fields['X']]
        
        return Batch(**batch_fields)


    