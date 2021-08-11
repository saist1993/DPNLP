import torch
# from torchtext.data.utils import ngrams_iterator

__all__ = [
    'vocab_func',
    'totensor',
    'ngrams_func',
    'sequential_transforms'
]


def ngrams_iterator(token_list, ngrams):
    """Return an iterator that yields the given tokens and their ngrams.

    Args:
        token_list: A list of tokens
        ngrams: the number of ngrams.

    Examples:
        >>> token_list = ['here', 'we', 'are']
        >>> list(ngrams_iterator(token_list, 2))
        >>> ['here', 'here we', 'we', 'we are', 'are']
    """

    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])

    for x in token_list:
        yield x
    for n in range(2, ngrams + 1):
        for x in _get_ngrams(n):
            yield ' '.join(x)

def vocab_func(vocab):
    def func(tok_iter):
        return [vocab[tok] for tok in tok_iter]

    return func


def totensor(dtype):
    def func(ids_list):
        return torch.tensor(ids_list).to(dtype)

    return func


def ngrams_func(ngrams):
    def func(token_list):
        return list(ngrams_iterator(token_list, ngrams))

    return func


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


class TextClassificationDataset(torch.utils.data.Dataset):
    """Defines an abstract text classification datasets.
       Currently, we only support the following datasets:

           - AG_NEWS
           - SogouNews
           - DBpedia
           - YelpReviewPolarity
           - YelpReviewFull
           - YahooAnswers
           - AmazonReviewPolarity
           - AmazonReviewFull
    """

    def __init__(self, data, vocab, transforms):
        """Initiate text-classification dataset.

        Assumption is that the first element in the list i.e. 0th is the label
        Args:
            data: a list of label and text tring tuple. label is an integer.
                [(label1, text1), (label2, text2), (label2, text3)]
            vocab: Vocabulary object used for dataset.
            transforms: a tuple of label and text string transforms.
        """

        super(TextClassificationDataset, self).__init__()
        self.data = data
        self.vocab = vocab
        self.transforms = transforms  # (label_transforms, tokens_transforms)
        # can I do some static mechanism to find all y and g;s

    def __getitem__(self, i):
        label = self.data[i][0]
        txt = self.data[i][1]
        # return (self.transforms[0](label), self.transforms[1](txt))

        final_data = []
        for data, transformation in zip(self.data[i], self.transforms):
            final_data.append(transformation(data))

        return tuple(final_data)

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        labels = []
        for item in self.data:
            label = item[0]
            labels.append(self.transforms[0](label))
        return set(labels)

    def get_vocab(self):
        return self.vocab


class CombinedIterator:
    # iterator assumes that itera and iterb has same length.
    def __init__(self, itera, iterb):
        self.itera = itera
        self.iterb = iterb
        assert len(self.itera) == len(self.iterb)

    def __iter__(self):

        for item1, item2 in zip(self.itera, self.iterb):
            new_item = {}
            for key in item1.keys():
                new_item[key] = torch.cat((item1[key], item2[key]), 0)

            yield new_item

    def __len__(self):
        return len(self.itera)  # len of itera is same as iterb
