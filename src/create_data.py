# All data reading manipulation for a dataset takes place here.
import copy
import torch
import numpy as np
from sklearn.datasets import load_svmlight_files
from sklearn.preprocessing import StandardScaler
from mytorch.utils.goodies import *

# custom imports
import config
from utils.simple_classification_dataset import *
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
        self.supervised_da = params['supervised_da']

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
                source_file_test = location + self.source_domain + '_test.svmlight'
                target_file = location + self.target_domain + '_train.svmlight'
                target_test_file = location + self.target_domain + '_test.svmlight'
                xs, ys, xs_test, ys_test, xt, yt, xt_test, yt_test = load_svmlight_files([source_file, source_file_test,
                                                                                          target_file, target_test_file])
                ys, yt, yt_test, ys_test = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, yt_test, ys_test))
                return xs.A, ys, xs_test.A, ys_test, xt.A, yt, xt_test.A, yt_test  #
            except FileNotFoundError:
                continue

    def load_leave_one_out(self):
        target_x_train, target_y_train, target_x_test, target_y_test = self.load_file(self.target_domain)
        src_x_train, src_y_train = [], []
        src_x_test, src_y_test = [], []
        for domain_name in self.all_domain_names:
            if domain_name != self.target_domain:
                src_x, src_y, _src_x_test, _src_y_test = self.load_file(domain_name)
                src_x_train.append(src_x)
                src_y_train.append(src_y)
                src_x_test.append(_src_x_test)
                src_y_test.append(_src_y_test)

        # stack src_x_train, src_y_train to get one output.
        src_x_train = np.vstack(src_x_train)
        src_y_train = np.hstack(src_y_train)
        src_x_test = np.vstack(src_x_test)
        src_y_test = np.hstack(src_y_test)


        return src_x_train, src_y_train, src_x_test, src_y_test, target_x_train, target_y_train, target_x_test, target_y_test

    def run(self):
        if self.supervised_da:
            return self.run_kfold()
        else:
            return self.run_non_kfold()

    def run_non_kfold(self):
        xs, ys, xs_test, ys_test, xt, yt, xt_test, yt_test = self.load_method()
        '''
            It can be the case that xs and xt are not same shape i.e not of equal length. 
            Thus to make sure that they are of equal size, we will oversample from xt 
        '''
        no_examples_to_sample = len(xs) - len(xt)
        sampled_index = np.random.randint(xt.shape[0], size=no_examples_to_sample)
        xt = np.vstack([xt, xt[sampled_index, :]])
        yt = np.hstack([yt, yt[sampled_index]])


        # shuffling data
        shuffle_ind = np.random.permutation(len(xs))
        xs, ys, xt, yt = xs[shuffle_ind], ys[shuffle_ind], xt[shuffle_ind], yt[shuffle_ind]

        shuffle_ind = np.random.permutation(len(xs_test))
        xs_test, ys_test = xs_test[shuffle_ind], ys_test[shuffle_ind]

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
        # dummy valid data. We are going to use this validation set to check for leakage.
        dev_data = self.process_data(xs_test[:len(xt_test)], ys_test[:len(xt_test)], np.zeros_like(yt_test[:512]), vocab=vocab)

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
        other_meta_data['task'] = 'unsupervised_domain_adaptation'
        other_meta_data['dataset_name'] = self.dataset_name

        return vocab, number_of_labels, number_of_labels, iterators, other_meta_data # empty dict for other_meta_data

    def run_kfold(self):
        k_fold = 5
        xs, ys, xs_test, ys_test, xt, yt, _, _ = self.load_method()
        s_source = np.zeros_like(ys)


        sampling_percentage = 1.0/k_fold
        target_samples = xt.shape[0] # do note we don't use the test set. @TODO: verify that others don't either
        one_fold_length = int(sampling_percentage*target_samples)
        fold_indexes = [[i for i in range(int((i-1)*one_fold_length),int(i*one_fold_length)) ] for i in range(k_fold+1)][1:]
        sampling_indexes = []
        iterators = []
        vocab = {'<pad>': 1}  # no need of vocab in these dataset. It is there for code compatibility purposes.


        for i in range(k_fold):
            test_index = i
            if i == k_fold - 1:
                valid_index = 0
            else:
                valid_index = i + 1
            ignore_index = [test_index, valid_index]
            train_index = [i for i in range(k_fold) if i not in ignore_index]
            sampling_indexes.append([test_index, valid_index, train_index])


        for test_index, valid_index, train_index in sampling_indexes:
            _xt_valid,_yt_valid, _xt_test, _yt_test = xt[fold_indexes[valid_index]], yt[fold_indexes[valid_index]], xt[fold_indexes[test_index]], yt[fold_indexes[test_index]]
            _xt_train = np.vstack([xt[fold_indexes[index]] for index in train_index])
            _yt_train = np.hstack([yt[fold_indexes[index]] for index in train_index])
            s_target = np.ones_like(_yt_train)

            # now combine - for train
            train_X = np.vstack([_xt_train, xs])
            train_s = np.hstack([s_target, s_source])
            train_y = np.hstack([_yt_train, ys])

            # shuffle data
            shuffle_index = np.random.permutation(train_X.shape[0])
            train_X, train_y, train_s = train_X[shuffle_index], train_y[shuffle_index], train_s[shuffle_index]


            # setup dev and test
            dev_X, dev_y, dev_s = _xt_valid,_yt_valid, np.ones_like(_yt_valid)
            test_X, test_y, test_s = _xt_test, _yt_test, np.ones_like(_yt_test)

            train_data = self.process_data(train_X, train_y, train_s, vocab=vocab)
            dev_data = self.process_data(dev_X, dev_y, dev_s, vocab=vocab)
            test_data = self.process_data(test_X, test_y, test_s, vocab=vocab)

            train_iterator = torch.utils.data.DataLoader(train_data,
                                                         self.batch_size,
                                                         shuffle=False,
                                                         collate_fn=self.collate
                                                         )

            dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                       512,
                                                       shuffle=False,
                                                       collate_fn=self.collate
                                                       )

            test_iterator = torch.utils.data.DataLoader(test_data,
                                                        512,
                                                        shuffle=False,
                                                        collate_fn=self.collate
                                                        )

            iterator_set = {
                'train_iterator': train_iterator,
                'valid_iterator': dev_iterator,
                'test_iterator': test_iterator,
            }
            iterators.append(iterator_set)

        other_meta_data = {}
        other_meta_data['task'] = 'supervised_domain_adaptation'
        other_meta_data['k_fold'] = k_fold
        other_meta_data['is_k_fold'] = True
        other_meta_data['dataset_name'] = self.dataset_name

        number_of_labels = len(np.unique(yt))



        return vocab, number_of_labels, number_of_labels, iterators, other_meta_data






class MultiGroupSenSR:
    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name
        self.X, self.y, self.s, self.s_concat = get_adult_multigroups_data_sensr()
        self.train_split = .80

    def process_data(self, X,y,s, vocab):
        """raw data is assumed to be tokenized"""
        final_data = [(a,b,c) for a,b,c in zip(y,X,s)]


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

    def run(self):
        # Add shuffle capabilites later.
        dataset_size = self.X.shape[0] # examples*feature_size
        dev_index = int(self.train_split * dataset_size) - int(self.train_split * dataset_size * .10)

        # shuffle here when needed.
        test_index = int(self.train_split * dataset_size)
        number_of_labels = len(np.unique(self.y))

        train_X, train_y, train_s = \
            self.X[:dev_index, :], self.y[:dev_index], self.s[:dev_index]

        dev_X, dev_y, dev_s = \
            self.X[dev_index:test_index, :], self.y[dev_index:test_index],\
            self.s[dev_index:test_index]

        test_X, test_y, test_s = \
            self.X[test_index:, :], self.y[test_index:], self.s[test_index:]

        vocab = {'<pad>': 1}  # no need of vocab in these dataset. It is there for code compatibility purposes.

        train_data = self.process_data(train_X, train_y, train_s, vocab=vocab)
        dev_data = self.process_data(dev_X, dev_y, dev_s, vocab=vocab)
        test_data = self.process_data(test_X, test_y, test_s, vocab=vocab)

        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        iterators = [] # If it was k-fold. One could append k iterators here.
        iterator_set = {
            'train_iterator': train_iterator,
            'valid_iterator': dev_iterator,
            'test_iterator': test_iterator
        }
        iterators.append(iterator_set)


        # make multiple copies of the dataset for gender race consistence
        '''
            Essentially: gender - [0,1]
                       : race   - [0,1]
                       
                       Create 4 copies of the dataset such that combination of race and gender are present 
                       
                       Copy1 - gender=0, race=0
                       Copy2 - gender=0, race=1
                       Copy3 - gender=1, race=0
                       Copy4 - gender=1, race=1
                       
                       Now see how many times is the prediction in all the copies is same.
                       Note that ground truth does not matter in this case
                       
                       gender_index = 39
                       race_index = 40
        '''

        gender_index = 39
        race_index = 40

        # copy 1 gender = 0, race = 0
        test_X[:, gender_index] = 0.0
        test_X[:, race_index] = 0.0
        X_copy1 = copy.deepcopy(test_X)

        # copy 2 gender = 0, race = 1
        test_X[:, gender_index] = 0.0
        test_X[:, race_index] = 1.0
        X_copy2 = copy.deepcopy(test_X)

        # copy 3 gender = 1, race = 0
        test_X[:, gender_index] = 1.0
        test_X[:, race_index] = 0.0
        X_copy3 = copy.deepcopy(test_X)

        # copy 4 gender = 1, race = 0
        test_X[:, gender_index] = 1.0
        test_X[:, race_index] = 1.0
        X_copy4 = copy.deepcopy(test_X)

        test_copy1 = self.process_data(X_copy1, test_y, test_s, vocab=vocab)
        test_copy2 = self.process_data(X_copy2, test_y, test_s, vocab=vocab)
        test_copy3 = self.process_data(X_copy3, test_y, test_s, vocab=vocab)
        test_copy4 = self.process_data(X_copy4, test_y, test_s, vocab=vocab)

        test_copy1_iterator = torch.utils.data.DataLoader(test_copy1,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        test_copy2_iterator = torch.utils.data.DataLoader(test_copy2,
                                                          512,
                                                          shuffle=False,
                                                          collate_fn=self.collate
                                                          )

        test_copy3_iterator = torch.utils.data.DataLoader(test_copy3,
                                                          512,
                                                          shuffle=False,
                                                          collate_fn=self.collate
                                                          )

        test_copy4_iterator = torch.utils.data.DataLoader(test_copy4,
                                                          512,
                                                          shuffle=False,
                                                          collate_fn=self.collate
                                                          )




        other_meta_data = {}
        other_meta_data['task'] = 'multi_aux_classification'
        other_meta_data['s_concat'] = self.s_concat
        other_meta_data['dataset_name'] = self.dataset_name
        other_meta_data['gender_race_consistent'] = [test_copy1_iterator, test_copy2_iterator,
                                                     test_copy3_iterator, test_copy4_iterator]

        return vocab, number_of_labels, number_of_labels, iterators, other_meta_data  # empty dict for other_meta_data

class SimpleAdvDatasetReader():
    def __init__(self, dataset_name:str,**params):
        self.dataset_name = dataset_name.lower()
        self.batch_size = params['batch_size']
        self.fairness_iterator = params['fairness_iterator']
        self.train_split = .80
        self.is_fair_grad = params['fair_grad_newer']

        if 'celeb' in self.dataset_name:
            self.X, self.y, self.s = get_celeb_data()
        elif 'adult' in self.dataset_name and 'multigroup' not in self.dataset_name:
            self.X, self.y, self.s = get_adult_data()
        elif 'crime' in self.dataset_name:
            self.X, self.y, self.s = get_crimeCommunities_data()
        elif 'dutch' in self.dataset_name:
            self.X, self.y, self.s = get_dutch_data()
        elif 'compas' in self.dataset_name:
            self.X, self.y, self.s = get_compas_data()
        elif 'german' in self.dataset_name:
            self.X, self.y, self.s = get_german_data()
        elif 'adult' in self.dataset_name and 'multigroup' in self.dataset_name:
            self.X, self.y, self.s = get_adult_multigroups_data()
        elif 'gaussian' in self.dataset_name:
            raise NotImplementedError
            # self.X, self.y, self.s = drh.get_gaussian_data(50000)
        else:
            raise NotImplementedError

        # converting all -1,1 -> 0,1
        self.y = (self.y+1)/2

        if len(np.unique(self.s)) == 2 and -1 in np.unique(self.s):
            self.s = (self.s + 1) / 2


    def process_data(self, X,y,s, vocab):
        """raw data is assumed to be tokenized"""

        final_data = [(a,b,c) for a,b,c in zip(y,X,s)]


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


    def run(self):

        dataset_size = self.X.shape[0] # examples*feature_size
        # the dataset is shuffled so as to get a unique test set for each seed.
        index = np.random.permutation(dataset_size)
        self.X, self.y, self.s = self.X[index], self.y[index], self.s[index]
        test_index = int(self.train_split*dataset_size)

        if self.is_fair_grad:
            dev_index = int(self.train_split*dataset_size) - int(self.train_split*dataset_size*.25)
        else:
            dev_index = int(self.train_split * dataset_size) - int(self.train_split * dataset_size * .10)

        number_of_labels = len(np.unique(self.y))
        number_of_aux_labels = len(np.unique(self.s))

        train_X, train_y, train_s = self.X[:dev_index,:], self.y[:dev_index], self.s[:dev_index]
        dev_X, dev_y, dev_s = self.X[dev_index:test_index, :], self.y[dev_index:test_index], self.s[dev_index:test_index]
        test_X, test_y, test_s = self.X[test_index:, :], self.y[test_index:], self.s[test_index:]

        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        dev_X = scaler.transform(dev_X)
        test_X = scaler.transform(test_X)





        vocab = {'<pad>':1} # no need of vocab in these dataset. It is there for code compatibility purposes.

        train_data = self.process_data(train_X,train_y,train_s, vocab=vocab)
        dev_data = self.process_data(dev_X,dev_y,dev_s, vocab=vocab)
        test_data = self.process_data(test_X,test_y,test_s, vocab=vocab)


        fairness_data = \
            create_fairness_data(
                train_X, train_y, train_s, dev_X, dev_y,
                dev_s, self.process_data, vocab, self.fairness_iterator)

        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        fairness_iterator = torch.utils.data.DataLoader(fairness_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        iterators = []  # If it was k-fold. One could append k iterators here.
        iterator_set = {
            'train_iterator': train_iterator,
            'valid_iterator': dev_iterator,
            'test_iterator': test_iterator,
            'fairness_iterator': fairness_iterator # now this can be independent of the dev iterator.
        }
        iterators.append(iterator_set)

        other_meta_data = {}
        other_meta_data['task'] = 'simple_classification'
        other_meta_data['dataset_name'] = self.dataset_name


        return vocab, number_of_labels, number_of_aux_labels, iterators, other_meta_data

class EncodedBiasInBios():
    def __init__(self, dataset_name:str,**params):
        self.dataset_name = dataset_name.lower()
        self.batch_size = params['batch_size']
        self.fairness_iterator = params['fairness_iterator']

        # generalize this. No edit without generalizing it!

        data_location = [Path('../datasets/bias_in_bios'), '../../../storage/fair_nlp_dataset/data/bias_in_bios']

        for d in data_location:
            try:
                self.train, self.dev, self.test = pickle.load(open(d/Path('train.pickle'), 'rb')),\
                                                  pickle.load(open(d/Path('dev.pickle'), 'rb')),\
                                                  pickle.load(open(d/Path('test.pickle'), 'rb'))

                self.train_cls, self.dev_cls, self.test_cls = np.load(d/Path('train.pickle_bert_cls.npy')), \
                                                  np.load(d/Path('dev.pickle_bert_cls.npy')), \
                                                  np.load(d/Path('test.pickle_bert_cls.npy'))

                # self.train_cls, self.dev_cls, self.test_cls = np.load(d / Path('debiased_x_train.npy')), \
                #                                               np.load(d / Path('debiased_x_dev.npy')), \
                #                                               np.load(d / Path('debiased_x_test.npy'))
                break
            except FileNotFoundError:
                continue




    def process_data(self, X,y,s, vocab):
        """raw data is assumed to be tokenized"""

        final_data = [(a,b,c) for a,b,c in zip(y,X,s)]


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


    def run(self):

        all_profession = list(set([t['p'] for t in self.train]))
        # profession_to_id = {profession: index for index, profession in enumerate(all_profession)}
        # pickle.dump(open("../datasets/bias_in_bios/prof_to_id", "wb"), profession_to_id )
        try:
            profession_to_id = pickle.load(open("../datasets/bias_in_bios/profession_to_id.pickle", "rb"))
        except:
            profession_to_id = pickle.load(
                open("/home/gmaheshwari/storage/fair_nlp_dataset/data/bias_in_bios/profession_to_id.pickle", "rb"))

        train_y = [profession_to_id[t['p']] for t in self.train]
        test_y = [profession_to_id[t['p']] for t in self.test]
        dev_y = [profession_to_id[t['p']] for t in self.dev]

        # gender
        gender = {'f':0, 'm':1}
        train_s = [gender[t['g']] for t in self.train]
        test_s = [gender[t['g']] for t in self.test]
        dev_s = [gender[t['g']] for t in self.dev]


        number_of_labels = len(np.unique(train_y))

        train_X, train_y, train_s = self.train_cls, train_y, train_s
        dev_X, dev_y, dev_s = self.dev_cls, dev_y, dev_s
        test_X, test_y, test_s = self.test_cls, test_y, test_s


        # scaler = StandardScaler().fit(train_X)
        # train_X = scaler.transform(train_X)
        # dev_X = scaler.transform(dev_X)
        # test_X = scaler.transform(test_X)





        vocab = {'<pad>':1} # no need of vocab in these dataset. It is there for code compatibility purposes.

        train_data = self.process_data(train_X,train_y,train_s, vocab=vocab)
        dev_data = self.process_data(dev_X,dev_y,dev_s, vocab=vocab)
        test_data = self.process_data(test_X,test_y,test_s, vocab=vocab)


        fairness_data = \
            create_fairness_data(
                train_X, train_y, train_s, dev_X, dev_y,
                dev_s, self.process_data, vocab, self.fairness_iterator)

        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        fairness_iterator = torch.utils.data.DataLoader(fairness_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        iterators = []  # If it was k-fold. One could append k iterators here.
        iterator_set = {
            'train_iterator': train_iterator,
            'valid_iterator': dev_iterator,
            'test_iterator': test_iterator,
            'fairness_iterator': fairness_iterator # now this can be independent of the dev iterator.
        }
        iterators.append(iterator_set)

        other_meta_data = {}
        other_meta_data['task'] = 'simple_classification'
        other_meta_data['dataset_name'] = self.dataset_name


        return vocab, number_of_labels, number_of_labels, iterators, other_meta_data


class EncodedEmoji:
    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name
        self.n = 100000 # https://github.com/HanXudong/Diverse_Adversaries_for_Mitigating_Bias_in_Training/blob/b5b4c99ada17b3c19ab2ae8789bb56058cb72643/scripts_deepmoji.py#L270
        self.folder_location = '../datasets/deepmoji'
        try:
            self.ratio = params['ratio_of_pos_neg']
        except:
            self.ratio = 0.8 # this the default in https://arxiv.org/pdf/2101.10001.pdf
        self.batch_size = params['batch_size']
        self.fairness_iterator = params['fairness_iterator']

    def read_data_file(self, input_file: str):
        vecs = np.load(input_file)

        np.random.shuffle(vecs)

        return vecs[:40000], vecs[40000:42000], vecs[42000:44000]

    def process_data(self, X,y,s, vocab):
        """raw data is assumed to be tokenized"""
        final_data = [(a,b,c) for a,b,c in zip(y,X,s)]


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

    def run(self):

        try:

            train_pos_pos, dev_pos_pos, test_pos_pos = self.read_data_file(f"{self.folder_location}/pos_pos.npy")
            train_pos_neg, dev_pos_neg, test_pos_neg = self.read_data_file(f"{self.folder_location}/pos_neg.npy")
            train_neg_pos, dev_neg_pos, test_neg_pos = self.read_data_file(f"{self.folder_location}/neg_pos.npy")
            train_neg_neg, dev_neg_neg, test_neg_neg = self.read_data_file(f"{self.folder_location}/neg_neg.npy")

        except:

            self.folder_location = '/home/gmaheshwari/storage/fair_nlp_dataset/data/deepmoji'
            train_pos_pos, dev_pos_pos, test_pos_pos = self.read_data_file(f"{self.folder_location}/pos_pos.npy")
            train_pos_neg, dev_pos_neg, test_pos_neg = self.read_data_file(f"{self.folder_location}/pos_neg.npy")
            train_neg_pos, dev_neg_pos, test_neg_pos = self.read_data_file(f"{self.folder_location}/neg_pos.npy")
            train_neg_neg, dev_neg_neg, test_neg_neg = self.read_data_file(f"{self.folder_location}/neg_neg.npy")


        n_1 = int(self.n * self.ratio / 2)
        n_2 = int(self.n * (1 - self.ratio) / 2)

        fnames = ['pos_pos.npy', 'pos_neg.npy', 'neg_pos.npy', 'neg_neg.npy']
        main_labels = [1, 1, 0, 0]
        protected_labels = [1, 0, 1, 0]
        ratios = [n_1, n_2, n_2, n_1]
        data = [train_pos_pos, train_pos_neg, train_neg_pos, train_neg_neg]

        X_train, y_train, s_train = [], [], []

        # loading data for train

        for data_file, main_label, protected_label, ratio in zip(data, main_labels, protected_labels, ratios):
            X_train = X_train + list(data_file[:ratio])
            y_train = y_train + [main_label] * len(data_file[:ratio])
            s_train = s_train + [protected_label] * len(data_file[:ratio])


        X_dev, y_dev, s_dev = [], [], []
        for data_file, main_label, protected_label in zip([dev_pos_pos, dev_pos_neg, dev_neg_pos, dev_neg_neg]
                , main_labels, protected_labels):
            X_dev = X_dev + list(data_file)
            y_dev = y_dev + [main_label] * len(data_file)
            s_dev = s_dev + [protected_label] * len(data_file)


        X_test, y_test, s_test = [], [], []
        for data_file, main_label, protected_label in zip([test_pos_pos, test_pos_neg, test_neg_pos, test_neg_neg]
                , main_labels, protected_labels):
            X_test = X_test + list(data_file)
            y_test = y_test + [main_label] * len(data_file)
            s_test = s_test + [protected_label] * len(data_file)


        X_train, y_train, s_train = np.asarray(X_train), np.asarray(y_train), np.asarray(s_train)
        X_dev, y_dev, s_dev = np.asarray(X_dev), np.asarray(y_dev), np.asarray(s_dev)
        X_test, y_test, s_test = np.asarray(X_test), np.asarray(y_test), np.asarray(s_test)

        # scaler = StandardScaler().fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_dev = scaler.transform(X_dev)
        # X_test = scaler.transform(X_test)




        all_x = [[a, b] for a, b in zip(y_train, s_train)]
        new_stuff = {}
        for i in all_x:
            try:
                new_stuff[str(i)] = new_stuff[str(i)] + 1
            except:
                new_stuff[str(i)] = 1

        print(new_stuff)

        vocab = {'<pad>':1} # no need of vocab in these dataset. It is there for code compatibility purposes.
        number_of_labels = 2

        # shuffling data
        shuffle_train_index = np.random.permutation(len(X_train))
        X_train, y_train, s_train = X_train[shuffle_train_index], y_train[shuffle_train_index], s_train[shuffle_train_index]

        shuffle_dev_index = np.random.permutation(len(X_dev))
        X_dev, y_dev, s_dev = X_dev[shuffle_dev_index], y_dev[shuffle_dev_index], s_dev[
            shuffle_dev_index]

        shuffle_test_index = np.random.permutation(len(X_test))
        X_test, y_test, s_test = X_test[shuffle_test_index], y_test[shuffle_test_index], s_test[
            shuffle_test_index]

        train_data = self.process_data(X_train,y_train,s_train, vocab=vocab)
        dev_data = self.process_data(X_dev,y_dev,s_dev, vocab=vocab)
        test_data = self.process_data(X_test,y_test,s_test, vocab=vocab)

        fairness_data = \
            create_fairness_data(
                X_train, y_train, s_train, X_dev, y_dev,
                s_dev, self.process_data, vocab, self.fairness_iterator)


        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        fairness_iterator = torch.utils.data.DataLoader(fairness_data,
                                                        512,
                                                        shuffle=False,
                                                        collate_fn=self.collate
                                                        )

        iterators = []  # If it was k-fold. One could append k iterators here.
        iterator_set = {
            'train_iterator': train_iterator,
            'valid_iterator': dev_iterator,
            'test_iterator': test_iterator,
            'fairness_iterator': fairness_iterator  # now this can be independent of the dev iterator.
        }
        iterators.append(iterator_set)

        other_meta_data = {}
        other_meta_data['task'] = 'simple_classification'
        other_meta_data['dataset_name'] = self.dataset_name

        return vocab, number_of_labels, number_of_labels, iterators, other_meta_data


class EncodedDpNLP:
    """Implements a set of dataset used by Differentially Private Representation for NLP: Formal Guarantee and An Empirical Study on Privacy and Fairness"""
    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name
        if self.dataset_name == 'blog':
            self.index_for_s = 0
        elif self.dataset_name == 'blog_v2':
            self.index_for_s = 1
        else:
            raise NotImplementedError
        if 'blog' in self.dataset_name: # blog dataset
            self.file_name = [] # @TODO: find a location and save it.
            self.file_name.append('../datasets/dpnlp/encoded_data/blog.pkl')
            self.file_name.append('/home/gmaheshwari/storage/dpnlp/encoded_data/blog.pkl')
        else:
            raise NotImplementedError

        self.fairness_iterator = params['fairness_iterator']

    def process_data(self, X,y,s, vocab):
        """raw data is assumed to be tokenized"""
        final_data = [(a,b,c) for a,b,c in zip(y,X,s)]


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

    def run(self):

        try:
            data = pickle.load(open(self.file_name[0], 'rb'))
        except FileNotFoundError:
            data = pickle.load(open(self.file_name[1], 'rb'))

        y_train = np.asarray([int(d.label) for d in data.get_train_examples()])
        s_train = np.asarray([int(d.aux_label[self.index_for_s]) for d in data.get_train_examples()])
        X_train = np.asarray(data.get_train_encoding())

        y_dev = np.asarray([int(d.label) for d in data.get_dev_examples()])
        s_dev = np.asarray([int(d.aux_label[self.index_for_s]) for d in data.get_dev_examples()])
        X_dev = np.asarray(data.get_dev_encoding())

        y_test = np.asarray([int(d.label) for d in data.get_test_examples()])
        s_test = np.asarray([int(d.aux_label[self.index_for_s]) for d in data.get_test_examples()])
        X_test = np.asarray(data.get_test_encoding())

        vocab = {'<pad>':1} # no need of vocab in these dataset. It is there for code compatibility purposes.
        number_of_labels = len(data.get_labels())


        # shuffling data
        shuffle_train_index = np.random.permutation(len(X_train))
        X_train, y_train, s_train = X_train[shuffle_train_index], y_train[shuffle_train_index], s_train[shuffle_train_index]

        shuffle_dev_index = np.random.permutation(len(X_dev))
        X_dev, y_dev, s_dev = X_dev[shuffle_dev_index], y_dev[shuffle_dev_index], s_dev[
            shuffle_dev_index]

        shuffle_test_index = np.random.permutation(len(X_test))
        X_test, y_test, s_test = X_test[shuffle_test_index], y_test[shuffle_test_index], s_test[
            shuffle_test_index]

        if self.dataset_name == 'blog':

            mask = np.logical_and(y_train == 0.0, s_train == 0.0)
            extra_example_y,extra_example_X,extra_example_s  = [y_train[mask][0]], [X_train[mask][0]] , [s_train[mask][0]]

            y_test = np.hstack([y_test, np.asarray(extra_example_y)])
            s_test = np.hstack([s_test, np.asarray(extra_example_s)])
            X_test = np.vstack([X_test, np.asarray(extra_example_X)])


            y_dev = np.hstack([y_dev, np.asarray(extra_example_y)])
            s_dev = np.hstack([s_dev , np.asarray(extra_example_s)])
            X_dev = np.vstack([X_dev, np.asarray(extra_example_X)])

            mask = np.logical_and(y_dev == 0.0, s_dev == 1.0)
            extra_example_y, extra_example_X, extra_example_s = [y_dev[mask][0]], [X_dev[mask][0]], [s_dev[mask][0]]

            y_test = np.hstack([y_test, np.asarray(extra_example_y)])
            s_test = np.hstack([s_test, np.asarray(extra_example_s)])
            X_test = np.vstack([X_test, np.asarray(extra_example_X)])

            mask = np.logical_and(y_dev == 8.0, s_dev == 1.0)
            extra_example_y, extra_example_X, extra_example_s = [y_dev[mask][0]], [X_dev[mask][0]], [s_dev[mask][0]]

            y_test = np.hstack([y_test, np.asarray(extra_example_y)])
            s_test = np.hstack([s_test, np.asarray(extra_example_s)])
            X_test = np.vstack([X_test, np.asarray(extra_example_X)])

        train_data = self.process_data(X_train,y_train,s_train, vocab=vocab)
        dev_data = self.process_data(X_dev,y_dev,s_dev, vocab=vocab)
        test_data = self.process_data(X_test,y_test,s_test, vocab=vocab)

        fairness_data = \
            create_fairness_data(
                X_train, y_train, s_train, X_dev, y_dev,
                s_dev, self.process_data, vocab, self.fairness_iterator)


        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        fairness_iterator = torch.utils.data.DataLoader(fairness_data,
                                                        512,
                                                        shuffle=False,
                                                        collate_fn=self.collate
                                                        )

        iterators = []  # If it was k-fold. One could append k iterators here.
        iterator_set = {
            'train_iterator': train_iterator,
            'valid_iterator': dev_iterator,
            'test_iterator': test_iterator,
            'fairness_iterator': fairness_iterator  # now this can be independent of the dev iterator.
        }
        iterators.append(iterator_set)

        other_meta_data = {}
        other_meta_data['task'] = 'simple_classification'
        other_meta_data['dataset_name'] = self.dataset_name
        other_meta_data['index_for_s'] = self.index_for_s

        return vocab, number_of_labels, number_of_labels, iterators, other_meta_data



def generate_data_iterators(dataset_name:str, **kwargs):


    if "amazon" in dataset_name.lower():
        dataset_creator = DomainAdaptationAmazon(dataset_name=dataset_name, **kwargs)
        vocab, number_of_labels, number_of_aux_labels, iterators, other_meta_data = dataset_creator.run()
    elif "adult_multigroup_sensr" in dataset_name:
        dataset_creator = MultiGroupSenSR(dataset_name=dataset_name, **kwargs)
        vocab, number_of_labels, number_of_aux_labels, iterators, other_meta_data = dataset_creator.run()
    elif dataset_name.lower() in "_".join(['celeb', 'crime', 'dutch', 'compas', 'german', 'adult', 'gaussian','adult', 'multigroups']):
        dataset_creator = SimpleAdvDatasetReader(dataset_name=dataset_name, **kwargs)
        vocab, number_of_labels, number_of_aux_labels, iterators, other_meta_data = dataset_creator.run()
    elif  dataset_name.lower() in "_".join(['blog', 'blog_v2']):
        dataset_creator = EncodedDpNLP(dataset_name=dataset_name, **kwargs)
        vocab, number_of_labels, number_of_aux_labels, iterators, other_meta_data = dataset_creator.run()
    elif dataset_name.lower() == 'encoded_emoji':
        dataset_creator = EncodedEmoji(dataset_name=dataset_name, **kwargs)
        vocab, number_of_labels, number_of_aux_labels, iterators, other_meta_data = dataset_creator.run()
    elif dataset_name.lower() == 'encoded_bias_in_bios':
        dataset_creator = EncodedBiasInBios(dataset_name=dataset_name, **kwargs)
        vocab, number_of_labels, number_of_aux_labels, iterators, other_meta_data = dataset_creator.run()
    else:
        raise CustomError("No such dataset")


    return vocab, number_of_labels, number_of_aux_labels, iterators, other_meta_data


def create_fairness_data_old(train_X, train_y, train_s, dev_X, dev_y, dev_s, process_data, vocab, method):
    sampling_percentage = .20
    if method.lower() == 'train':
        return process_data(train_X, train_y, train_s, vocab=vocab)
    elif method.lower() == 'custom_1':
        # sample 10% of train and all of dev
        sampled_index = np.random.randint(train_X.shape[0], size=int(train_X.shape[0] * .10))
        fairness_X, fairness_y, fairness_s = np.vstack([train_X[sampled_index], dev_X]) \
            , np.hstack([train_y[sampled_index], dev_y]), np.hstack([train_s[sampled_index], dev_s])
        return process_data(fairness_X, fairness_y, fairness_s, vocab=vocab)
    elif method.lower == 'custom_2':
        # 20% of the train samples
        sampled_index = np.random.randint(train_X.shape[0], size=int(train_X.shape[0] * sampling_percentage))
        fairness_X, fairness_y, fairness_s = train_X[sampled_index], train_y[sampled_index], train_s[sampled_index]
        return process_data(fairness_X, fairness_y, fairness_s, vocab=vocab)
    elif method.lower == 'custom_3':
        # just validation
        return process_data(dev_X, dev_y, dev_s, vocab=vocab)
    elif method.lower == 'custom_4':
        # 20% of the randomly sampled data
        total_size = train_X.shape[0] + dev_X.shape[0]
        sampled_index = np.random.randint(total_size, size=int(total_size * sampling_percentage))
        fairness_X, fairness_y, fairness_s = np.vstack([train_X, dev_X])[sampled_index] \
            , np.hstack([train_y, dev_y])[sampled_index], np.hstack([train_s, dev_s])[sampled_index]
        return process_data(fairness_X, fairness_y, fairness_s, vocab=vocab)
    else:
        raise NotImplementedError


def create_fairness_data(train_X, train_y, train_s, dev_X, dev_y, dev_s, process_data, vocab, method):
    sampling_percentage = .20
    total_size = train_X.shape[0] + dev_X.shape[0]
    no_examples_to_sample = int(total_size*sampling_percentage)
    # assert dev_X.shape[0] >= no_examples_to_sample

    if method.lower() == 'train':
        return process_data(train_X, train_y, train_s, vocab=vocab)
    elif method.lower() == 'custom_1':
        # sample from train
        sampled_index = np.random.randint(train_X.shape[0], size=no_examples_to_sample)
        fairness_X, fairness_y, fairness_s = train_X[sampled_index], train_y[sampled_index], train_s[sampled_index]
        return process_data(fairness_X, fairness_y, fairness_s, vocab=vocab)
    elif method.lower() == 'custom_2':
        # sample from valid
        sampled_index = np.random.randint(dev_X.shape[0], size=no_examples_to_sample)
        fairness_X, fairness_y, fairness_s = dev_X[sampled_index], dev_y[sampled_index], dev_s[sampled_index]
        return process_data(fairness_X, fairness_y, fairness_s, vocab=vocab)
    elif method.lower() == 'custom_3':
        # sample randomly from train + valid
        sampled_index = np.random.randint(total_size, size=no_examples_to_sample)
        fairness_X, fairness_y, fairness_s = np.vstack([train_X, dev_X])[sampled_index] \
            , np.hstack([train_y, dev_y])[sampled_index], np.hstack([train_s, dev_s])[sampled_index]
        return process_data(fairness_X, fairness_y, fairness_s, vocab=vocab)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    dataset_name = 'amazon_electronics'
    params = {
        'batch_size': 64,
        'fairness_iterator': 'custom_3',
        'fair_grad_newer': False
    }

    dataset = EncodedBiasInBios(dataset_name, **params)
    dataset.run()