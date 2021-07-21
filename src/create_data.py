# All data reading manipulation for a dataset takes place here.

import torch
import numpy as np
from sklearn.datasets import load_svmlight_files



class DomainAdaptationAmazon:
    """
    In a manner a more unique task when compared to the above ones. However, once the test/train/valid have
    been appropriately calibrated. THis would be straightforward.
    """

    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name  # dataset_name amazon_sourceDataset_targetDataset
        # source_name = 'dvd'; target_name = 'electronics'; dataset_name = amazon_dvd_electronics
        self.source_name, self.target_name = dataset_name.split('_')[1:]
        self.file_location = []
        self.file_location.append('../data/amazon/')
        self.file_location.append('/home/gmaheshwari/storage/amazon/')

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

        return labels, input, lengths, aux

    def load_amazon(self, source_name, target_name):
        try:
            location = self.file_location[0]
            source_file = location + source_name + '_train.svmlight'
            target_file = location + target_name + '_train.svmlight'
            test_file = location + target_name + '_test.svmlight'
            xs, ys, xt, yt, xt_test, yt_test = load_svmlight_files([source_file, target_file, test_file])
        except FileNotFoundError:
            location = self.file_location[1]
            print(f"in except {location}")
            source_file = location + source_name + '_train.svmlight'
            target_file = location + target_name + '_train.svmlight'
            test_file = location + target_name + '_test.svmlight'
            xs, ys, xt, yt, xt_test, yt_test = load_svmlight_files([source_file, target_file, test_file])

        ys, yt, yt_test = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, yt_test))

        return xs.A, ys, xt.A, yt, xt_test.A, yt_test  # .A converts sparse matrix to dense matrix.

    def run(self):
        xs, ys, xt, yt, xt_test, yt_test = self.load_amazon(self.source_name, self.target_name)
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
        dev_iterator = torch.utils.data.DataLoader(test_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        number_of_labels = len(np.unique(yt))

        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator, number_of_labels