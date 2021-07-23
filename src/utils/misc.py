import re
import scipy
import torch
from tqdm.auto import tqdm

def calculate_accuracy_classification(predictions, labels):
    top_predictions = predictions.argmax(1, keepdim = True)
    correct = top_predictions.eq(labels.view_as(top_predictions)).sum()
    accuracy = correct.float() / labels.shape[0]
    return accuracy

def calculate_accuracy_regression(predictions, labels):
    return scipy.stats.pearsonr(predictions.squeeze().detach().cpu().numpy(), labels.cpu().detach().cpu().numpy())[0]

def clean_text(text:str):
    """
    cleans text casing puntations and special characters. Removes extra space
    """
    text = re.sub('[^ a-zA-Z0-9]|unk', '', text)
    text = text.strip()
    return text

clean_text_functional = clean_text


def clean_text_tweet(text:str):
    return text.replace('#', '').replace('@', '')

def tester(text):
    print(f"this is the {text}")
    return text

def get_pretrained_embedding(initial_embedding, pretrained_vectors, vocab, device):
    pretrained_embedding = torch.FloatTensor(initial_embedding.weight.clone()).cpu().detach().numpy()

    # if device == 'cpu':
    #     pretrained_embedding = torch.FloatTensor(initial_embedding.weight.clone()).cpu().detach().numpy()
    # else:
    #     pretrained_embedding = torch.FloatTensor(initial_embedding.weight.clone()).cuda().detach().numpy()

    unk_tokens = []

    for idx, token in tqdm(enumerate(vocab.itos)):
        try:
            pretrained_embedding[idx] = pretrained_vectors[token]
        except KeyError:
            unk_tokens.append(token)

    pretrained_embedding = torch.from_numpy(pretrained_embedding).to(device)
    return pretrained_embedding, unk_tokens

def resolve_device(device = None) -> torch.device:
    """Resolve a torch.device given a desired device (string)."""
    if device is None or device == 'gpu':
        device = 'cuda'
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
        print('No cuda devices were available. The model runs on CPU')
    return device