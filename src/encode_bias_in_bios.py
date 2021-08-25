# a script primarly copied from https://github.com/shauli-ravfogel/nullspace_projection/blob/master/src/data/encode_bert_states.py
# reads bias in bios dataset and encodes it.

import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import BertModel, BertTokenizer


def read_data_file(input_file):
    """
    read the data file with a pickle format
    :param input_file: input path, string
    :return: the file's content
    """
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return data


def load_lm():
    """
    load bert's language model
    :return: the model and its corresponding tokenizer
    """
    model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return model, tokenizer


def tokenize(tokenizer, data):
    """
    Iterate over the data and tokenize it. Sequences longer than 512 tokens are trimmed.
    :param tokenizer: tokenizer to use for tokenization
    :param data: data to tokenize
    :return: a list of the entire tokenized data
    """
    tokenized_data = []
    for row in tqdm(data):
        tokens = tokenizer.encode(row['hard_text'], add_special_tokens=True)
        # keeping a maximum length of bert tokens: 512
        tokenized_data.append(tokens[:512])
    return tokenized_data


def encode_text(model, data):
    """
    encode the text
    :param model: encoding model
    :param data: data
    :return: two numpy matrices of the data:
                first: average of all tokens in each sentence
                second: cls token of each sentence
    """
    all_data_cls = []
    all_data_avg = []
    batch = []
    for row in tqdm(data):
        batch.append(row)
        input_ids = torch.tensor(batch)
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]
            all_data_avg.append(last_hidden_states.squeeze(0).mean(dim=0).numpy())
            all_data_cls.append(last_hidden_states.squeeze(0)[0].numpy())
        batch = []
    return np.array(all_data_avg), np.array(all_data_cls)


from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def batch_tokenize(tokenizer, data, batch_size):
    """
    Uses batch tokenizer to create batches of the dataset, instead of just one sentences at a time.
    :param data:
    :return:
    """

    batched_and_tokenized = []

    sentences = [d['hard_text'] for d in data]

    # break the sentences into batches
    batches = grouper(sentences,batch_size)

    # from the last batch remove None!
    for batch in tqdm(batches):
        # remove none
        batch = [i for i in batch if i is not None]
        batched_and_tokenized.append(tokenizer(batch, padding=True, truncation=True, return_tensors="pt"))

    return batched_and_tokenized


def masked_mean(vals: torch.Tensor, mask: torch.Tensor):
    """ vals (bs, sl, hdim), mask: (bs, sl) """
    seqlens = torch.sum(mask, dim=1).unsqueeze(1)   # (bs,1)
    masked_vals = vals * mask.unsqueeze(-1)
    return torch.sum(masked_vals, dim=1) / seqlens

def encode_text_batch(model, data, device):
    """
    encode the text
    :param model: encoding model
    :param data: data
    :return: two numpy matrices of the data:
                first: average of all tokens in each sentence
                second: cls token of each sentence
    """
    all_data_cls = []
    all_data_avg = []
    for batch in tqdm(tokens):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            last_hidden_states = model(**batch).last_hidden_state
            #

            all_data_avg.append(masked_mean(last_hidden_states, batch['attention_mask']).detach().cpu().numpy())
            all_data_cls.append(last_hidden_states[:,0,:].detach().cpu().numpy())
    return np.vstack(all_data_avg), np.vstack(all_data_cls)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', '-in_file', help="input file with exact path", type=str)
    parser.add_argument('--split', '-split', help="the split under consideration", type=str)

    args = parser.parse_args()
    in_file = Path(args.in_file)
    split = Path(args.split + '.pickle')

    print(f"file under consideration {in_file / split }")
    model, tokenizer = load_lm()
    data = read_data_file(in_file/split)

    print("tokenizing the text")
    # tokens = tokenize(tokenizer, data)
    tokens = batch_tokenize(tokenizer, data)

    print("encoding the text")
    avg_data, cls_data = encode_text_batch(model, tokens)

    print("saving the text")
    np.save(in_file/ Path(f'{str(split)}_bert_avg.npy'), avg_data)
    np.save(in_file/ Path(f'{str(split)}_bert_cls.npy'), cls_data)