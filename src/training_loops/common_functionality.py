import torch
from tqdm.auto import tqdm
from sklearn.svm import LinearSVC

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
    if method == 'svm':
        biased_classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)
        biased_classifier.fit(train_preds, train_labels) # remember aux labels
        leakage = biased_classifier.score(test_preds, test_labels) # remember aux labels
        return leakage
    else:
        raise NotImplementedError

def generate_predictions(model, iterator, device):
    all_preds = []
    with torch.no_grad():
        for items in tqdm(iterator):

            # setting up the device
            for key in items.keys():
                items[key] = items[key].to(device)

            items['gradient_reversal'] = False
            output = model(items)
            predictions = output['prediction']
            all_preds.append(predictions.argmax(1))

    # flattening all_preds
    all_preds = torch.cat(all_preds, out=torch.Tensor(len(all_preds), all_preds[0].shape[0])).to(device)
    return all_preds