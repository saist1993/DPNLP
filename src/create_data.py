# All data reading manipulation for a dataset takes place here.
import torch
import numpy as np
from sklearn.datasets import load_svmlight_files
from mytorch.utils.goodies import *

# custom imports
import config
from utils.iterator import *
from utils.iterator import CombinedIterator, TextClassificationDataset

class DomainAdaptationAmazon:
    """
    In a manner a more unique task when compared to the above ones. However, once the test/train/valid have
    been appropriately calibrated. THis would be straightforward.
    """

    def __init__(self, dataset_name, **params):
        '''

        :param dataset_name: amazon_sourceDomain_targetDomain
        If amazon_targetDataset then all other domains are treated as source domain.
        dataset_name amazon_sourceDataset_targetDataset
        source_name = 'dvd'; target_name = 'electronics'; dataset_name = amazon_dvd_electronics
        dataset_name: amazon_electronics
        target domain is electronics and source domains are all other domains.

        :param params: all other params.
        '''
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name

        if len(self.dataset_name.split("_")) == 3:
            self.source_domain, self.target_domain = dataset_name.split('_')[1:]
            self.load_method = self.load_one_src_domain
        else:
            self.target_domain = dataset_name.split('_')[1]
            self.load_method = self.load_leave_one_out

        self.file_location = config.dataset_location['amazon']
        self.all_domain_names = ['books', 'dvd', 'electronics', 'kitchen']

    def process_data(self, X, y, s, vocab):
        """raw data is assumed to be tokenized"""
        final_data = [(a, b, c) for a, b, c in zip(y, X, s)]

        label_transform = sequential_transforms()
        input_transform = sequential_transforms()
        aux_transform = sequential_transforms()

        transforms = (label_transform, input_transform, aux_transform)

        return TextClassificationDataset(final_data, vocab, transforms)

    def collate(self, batch):
        labels, input, aux = zip(*batch)

        labels = torch.LongTensor(labels)
        aux = torch.LongTensor(aux)
        lengths = torch.LongTensor([len(x) for x in input])
        input = torch.FloatTensor(input)

        input_data = {
            'labels': labels,
            'input': input,
            'lengths': lengths,
            'aux': aux
        }

        return input_data

    def load_file(self, domain_name):
        for location in self.file_location:
            try:
                source_file_train = location + domain_name + '_train.svmlight'
                source_file_test = location + domain_name + '_test.svmlight'
                x_train, y_train, x_test, y_test = load_svmlight_files([source_file_train, source_file_test])
                y_train, y_test = (np.array((y + 1) / 2, dtype=int) for y in (y_train, y_test))
                return x_train.A, y_train, x_test.A, y_test
            except FileNotFoundError:
                continue

    def load_one_src_domain(self):

        for location in self.file_location:
            try:
                source_file = location + self.source_domain + '_train.svmlight'
                target_file = location + self.target_domain + '_train.svmlight'
                test_file = location + self.target_domain + '_test.svmlight'
                xs, ys, xt, yt, xt_test, yt_test = load_svmlight_files([source_file, target_file, test_file])
                ys, yt, yt_test = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, yt_test))
                return xs.A, ys, xt.A, yt, xt_test.A, yt_test  #
            except FileNotFoundError:
                continue

    def load_leave_one_out(self):
        target_x_train, target_y_train, target_x_test, target_y_test = self.load_file(self.target_domain)
        src_x_train, src_y_train = [], []
        for domain_name in self.all_domain_names:
            if domain_name != self.target_domain:
                src_x, src_y, _, _ = self.load_file(domain_name)
                src_x_train.append(src_x)
                src_y_train.append(src_y)

        # stack src_x_train, src_y_train to get one output.
        src_x_train = np.vstack(src_x_train)
        src_y_train = np.vstack(src_y_train)
        return src_x_train, src_y_train, target_x_train, target_y_train, target_x_test, target_y_test


    def run(self):
        xs, ys, xt, yt, xt_test, yt_test = self.load_method()
        '''
            It can be the case that xs and xt are not same shape i.e not of equal length. 
            Thus to make sure that they are of equal size, we will oversample from xt 
        '''
        no_examples_to_sample = len(xs) - len(xt)
        sampled_index = np.random.randint(xt.shape[0], size=no_examples_to_sample)
        xt = np.vstack([xt, xt[sampled_index, :]])
        yt = np.vstack([yt, yt[sampled_index, :]])


        # shuffling data
        shuffle_ind = np.random.permutation(len(xs))
        xs, ys, xt, yt = xs[shuffle_ind], ys[shuffle_ind], xt[shuffle_ind], yt[shuffle_ind]

        # split the train data into validation and train
        '''
        As per the paper: Wasserstein Distance Guided Representation Learning for Domain Adaptation.

        We follow Long et al. (Transfer Feature Learning with Joint Distribution Adaptation) and evaluate all compared 
        approaches through grid search on the hyperparameter space, and
        report the best results of each approach.
            -src https://github.com/RockySJ/WDGRL/issues/5

        This implies that the hyper-params were choosen based on the test set. Thus for now our validation
        set is same as test set.  
        '''
        #
        vocab = {'<pad>': 1}  # no need of vocab in these dataset. It is there for code compatibility purposes.
        train_src_data = self.process_data(xs, ys, np.zeros_like(ys), vocab=vocab)  # src s=0
        train_target_data = self.process_data(xt, yt, np.ones_like(yt), vocab=vocab)  # target s=1
        test_data = self.process_data(xt_test, yt_test, np.ones_like(yt_test), vocab=vocab)
        # dummy valid data. A small set of test set so that no time is wasted on validation set.
        dev_data = self.process_data(xt_test[:512], yt_test[:512], np.ones_like(yt_test[:512]), vocab=vocab)

        train_src_iterator = torch.utils.data.DataLoader(train_src_data,
                                                         self.batch_size,
                                                         shuffle=False,
                                                         collate_fn=self.collate
                                                         )

        train_target_iterator = torch.utils.data.DataLoader(train_target_data,
                                                            self.batch_size,
                                                            shuffle=False,
                                                            collate_fn=self.collate
                                                            )

        train_iterator = CombinedIterator(train_src_iterator, train_target_iterator)

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        # dev/valid is same as test as we are optimizing over the test set. See comments above.
        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        number_of_labels = len(np.unique(yt))

        iterators = [] # If it was k-fold. One could append k iterators here.
        iterator_set = {
            'train_iterator': train_iterator,
            'valid_iterator': dev_iterator,
            'test_iterator': test_iterator
        }
        iterators.append(iterator_set)

        other_meta_data = {}
        other_meta_data['task'] = 'domain_adaptation'

        return vocab, number_of_labels, number_of_labels, iterators, {} # empty dict for other_meta_data



def generate_data_iterators(dataset_name:str, **kwargs):


    if "amazon" in dataset_name.lower():
        dataset_creator = DomainAdaptationAmazon(dataset_name=dataset_name, **kwargs)
        vocab, number_of_labels, number_of_aux_labels, iterators, other_meta_data = dataset_creator.run()
    else:
        raise CustomError("No such dataset")


    return vocab, number_of_labels, number_of_aux_labels, iterators, other_meta_data

if __name__ == '__main__':
    dataset_name = 'amazon_electronics'
    params = {
        'batch_size': 64,
    }

    dataset = DomainAdaptationAmazon(dataset_name, **params)
    dataset.run()