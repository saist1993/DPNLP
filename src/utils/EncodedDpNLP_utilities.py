import os
import pickle

class Example(object):
    def __init__(self, guid, text_a, label, meta, att):
        self.guid = guid
        self.text_a = text_a
        self.text_b = None
        self.label = label
        self.att = att
        self.aux_label = []
        self.meta = meta

        for no in range(att):
            if str(no) in meta:
                self.aux_label.append("1")
            else:
                self.aux_label.append("0")

class Data(object):
    '''
    #Author   : Zodiac
    Copied from: https://github.com/xlhex/dpnlp
    '''
    def __init__(self, data_path, att=2):

        sep = "="*20
        dataset = [[], [], []]
        split = 0
        labels = set()

        guid = 0
        with open(data_path) as f:
            for line in f:
                line = line.lstrip()
                if line.strip() == sep:
                    split += 1
                    guid = 0
                    continue
                label, text, meta = line.split("\t")
                dataset[split].append(Example(guid, text, label, meta.strip(), att))
                guid += 1

                labels.add(label)

        self.label_list = list(labels)
        self.dataset = dataset
        self.train_encoding = None
        self.test_encoding = None
        self.dev_encoding = None
        self.encoding_info = None

    def get_labels(self):
        return self.label_list

    def get_train_examples(self):
        return self.dataset[0]

    def get_dev_examples(self):
        return self.dataset[1]

    def get_test_examples(self):
        return self.dataset[2]

    def get_train_encoding(self):
        return self.train_encoding

    def get_dev_encoding(self):
        return self.dev_encoding

    def get_test_encoding(self):
        return self.test_encoding

    def set_train_encoding(self, encodings):
        self.train_encoding = encodings

    def set_dev_encoding(self, encodings):
        self.dev_encoding = encodings

    def set_test_encoding(self, encodings):
        self.test_encoding = encodings

class AG_data(Data):
    @classmethod
    def get_ag_data(cls, data_dir):
        data_path = os.path.join(data_dir, "ag_data.txt")
        return cls(data_path, att=5)

class Blog_data(Data):
    @classmethod
    def get_blog_data(cls, data_dir):
        data_path = os.path.join(data_dir, "blog_data.txt")
        return cls(data_path, att=2)

class TP_data(Data):
    @classmethod
    def get_tp_data(cls, data_dir):
        data_path = os.path.join(data_dir, "tp_us.txt")
        return cls(data_path, att=2)

class TPUK_data(Data):
    @classmethod
    def get_tp_data(cls, data_dir):
        data_path = os.path.join(data_dir, "tp_uk.txt")
        return cls(data_path, att=2)

def get_processors(data_dir):
    get_data = {"ag": lambda : AG_data.get_ag_data(data_dir),
                "bl": lambda : Blog_data.get_blog_data(data_dir),
                "tp": lambda : TP_data.get_tp_data(data_dir),
                "tpuk": lambda : TPUK_data.get_tp_data(data_dir),
                }

    return get_data

def load_custom_file(file_name):
    return pickle.load(open(file_name, 'rb'))