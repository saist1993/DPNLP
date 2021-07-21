import re
import scipy

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

def clean_text_tweet(text:str):
    return text.replace('#', '').replace('@', '')

def tester(text):
    print(f"this is the {text}")
    return text