import math
import torch
import traceback
import numpy as np
from tqdm.auto import tqdm
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def calculate_fairness_stuff(all_preds, y, s, fairness_score_function, device, other_metadata=None):
    """this need a lot more information. We need to pass along more data for doing several stuff."""

    total_no_main_classes, total_no_aux_classes = len(torch.unique(y)), len(torch.unique(s))
    scoring_function_params = {
        'device': device,
        'total_no_aux_classes': total_no_aux_classes,
        'total_no_main_classes': total_no_main_classes,
        'other_metadata': other_metadata
    }
    grms, group_fairness = fairness_score_function(all_preds, y, s, scoring_function_params)

    return grms, group_fairness

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_lekage_old(model, dev_iterator, test_iterator, device):

    def temp(model, iterator, device):
        all_hidden = []
        all_prediction = []
        s = []
        with torch.no_grad():
            for labels, text, lengths, aux in tqdm(iterator):
                text = text.to(device)
                aux = aux.to(device)
                predictions, adv_output, _,  hidden = model(text, lengths, return_hidden=True)




                if len(predictions) == 2:
                    all_hidden.append(hidden.detach().cpu())
                    all_prediction.append(predictions.detach().cpu())
                else:
                    all_hidden.append(hidden.detach().cpu())
                    all_prediction.append(predictions.detach().cpu())
                s.append(aux)



        # flattening all_preds
        s = torch.cat(s, out=torch.Tensor(len(s), s[0].shape[0])).detach().cpu().numpy()
        all_hidden = torch.cat(all_hidden, out=torch.Tensor(len(all_hidden), all_hidden[0].shape[0])).detach().cpu().numpy()
        all_prediction = torch.cat(all_prediction, out=torch.Tensor(len(all_prediction), all_prediction[0].shape[0])).detach().cpu().numpy()
        return all_hidden, s, all_prediction

    dev_preds, dev_aux, dev_logits = temp(model, dev_iterator, device)
    test_preds, test_aux, test_logits = temp(model, test_iterator, device)

    biased_classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)

    biased_classifier.fit(dev_preds, dev_aux)
    test_hidden_leakage = biased_classifier.score(test_preds, test_aux)

    biased_classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)
    biased_classifier.fit(dev_logits, dev_aux)
    test_logits_leakage = biased_classifier.score(test_logits, test_aux)

    return test_hidden_leakage, test_logits_leakage

def calculate_leakage(train_preds, train_labels, test_preds, test_labels, method='svm'):
    # train_preds = train_preds.detach().cpu().numpy()
    train_labels = train_labels.detach().cpu().numpy()
    # test_preds = test_preds.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    if method == 'svm':
        biased_classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)
        biased_classifier.fit(train_preds, train_labels) # remember aux labels
        leakage = biased_classifier.score(test_preds, test_labels) # remember aux labels
        return leakage
    elif method == 'sgd':
        biased_classifier = SGDClassifier(max_iter=1000, tol=1e-3)
        clf = make_pipeline(StandardScaler(), biased_classifier)
        clf.fit(train_preds, train_labels) # remember aux labels
        leakage = clf.score(test_preds, test_labels) # remember aux labels
        return leakage
    elif method == 'neural_network':
        biased_classifier = MLPClassifier(alpha=1, max_iter=1000)
        clf = make_pipeline(StandardScaler(), biased_classifier)
        clf.fit(train_preds, train_labels) # remember aux labels
        leakage = clf.score(test_preds, test_labels) # remember aux labels
        return leakage
    elif method == 'mdl':
        # Use MDL
        mdl = MDL(train_preds, train_labels, test_preds, test_labels)
        leakage = mdl.get_score()
        return leakage
    else:
        raise NotImplementedError

def generate_predictions(model, iterator, device):
    all_preds = []
    fairness_all_aux, fairness_all_labels = [], []

    with torch.no_grad():
        for items in tqdm(iterator):

            # setting up the device
            for key in items.keys():
                items[key] = items[key].to(device)

            fairness_all_aux.append(items['aux'])  # aux label.
            fairness_all_labels.append(items['labels'])  # main task label.

            items['gradient_reversal'] = False
            output = model(items)
            predictions = output['prediction']
            all_preds.append(predictions.argmax(1))

    # flattening all_preds
    all_preds = torch.cat(all_preds, out=torch.Tensor(len(all_preds), all_preds[0].shape[0])).to(device)

    fairness_all_aux = torch.cat(fairness_all_aux, out=torch.Tensor(len(fairness_all_aux), fairness_all_aux[0].shape[0])).to(device)
    fairness_all_labels = torch.cat(fairness_all_labels, out=torch.Tensor(len(fairness_all_labels), fairness_all_labels[0].shape[0])).to(device)
    total_no_aux_classes, total_no_main_classes = len(torch.unique(fairness_all_aux)), len(torch.unique(fairness_all_labels))


    return all_preds, fairness_all_aux, fairness_all_labels, total_no_aux_classes, total_no_main_classes


def custom_generate_prediction(output, items, device):
    all_preds = output['prediction'].argmax(1)
    # fairness_all_preds = torch.cat(all_preds, out=torch.Tensor(len(all_preds), all_preds[0].shape[0])).to(device)
    fairness_all_aux = items['aux']
    fairness_all_labels = items['labels']
    total_no_aux_classes, total_no_main_classes = len(torch.unique(fairness_all_aux)), len(
        torch.unique(fairness_all_labels))

    return all_preds, fairness_all_aux, fairness_all_labels, total_no_aux_classes, total_no_main_classes


class MDL:
    # This code is from repo: https://github.com/brcsomnath/adversarial-scrubber
    def __init__(self, train_preds, train_labels, test_preds, test_labels):
        super(MDL, self).__init__()

        self.dataset_preds = np.vstack((train_preds, test_preds))
        self.dataset_labels = np.hstack((train_labels, test_labels))
        self.index = np.arange(len(self.dataset_preds))
        self.num_labels = len(np.unique(self.dataset_labels))
        np.random.shuffle(self.index)

        ratios = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25, 0.5, 1]
        self.all_datasets = []

        for r in ratios:
            self.all_datasets.append(self.index[:int(r * len(self.index))])

    def get_score(self):
        score = len(self.all_datasets[0]) * math.log(self.num_labels, 2)

        print("Computing MDL ...")
        for i, dataset in tqdm(enumerate(self.all_datasets[:-1]), total=len(self.all_datasets[:-1])):
            X_train = self.dataset_preds[dataset]
            Y_train = self.dataset_labels[dataset]


            clf = MLPClassifier(alpha=1, max_iter=800)
            clf.fit(X_train, Y_train)

            next_dataset = self.all_datasets[i + 1]
            X_test = self.dataset_preds[next_dataset]
            Y_test = self.dataset_labels[next_dataset]

            Y_pred = clf.predict_proba(X_test)

            for y_gold, y_pred in zip(Y_test, Y_pred):
                try:
                    score -= math.log(y_pred[int(y_gold)], 2)
                except:
                    traceback.print_exc()
                    pass
        return score / 1024