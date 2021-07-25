import math
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def equal_odds(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """

    :param preds: output/prediction of the model
    :param y: actual/ground/gold label
    :param s: aux output/ protected demographic attribute
    :param epsilon:
    :return:
    """

    unique_classes = torch.sort(torch.unique(y))[0] # For example: [doctor, nurse, engineer]
    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0] # For example: [Male, Female]
    group_fairness = {} # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])
    '''
    it will have a structure of 
    {
        'doctor': {
            'm' : 0.5, 
            'f' : 0.6
            }, 
        'model': {
        'm': 0.5,
        'f': 0.7,
        }
    '''
    for uc in unique_classes: # iterating over each class say: uc=doctor for the first iteration
        group_fairness[uc] = {}
        positive_rate = torch.mean((preds[y==uc] == uc).float()) # prob(pred=doctor/y=doctor)
        for group in unique_groups: # iterating over each group say: group=male for the firt iteration
            mask_pos = torch.logical_and(y==uc, s==group) # find instances with y=doctor and s=male
            g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
            g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            fairness[mask_pos] = g_fairness_pos
            group_fairness[uc][group] = g_fairness_pos
            fairness_lookup[int(uc.item()),int(group.item())] = g_fairness_pos


    return group_fairness, fairness_lookup


def demographic_parity(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """
    We assume y to be 0,1  that is y can only take 1 or 0 as the input.

    y = 1 is considered to be a positive class while y=0 is considered to be a negative class. This is
    unlike other methods where y=-1 is considered to be negative class.

    In demographic parity the fairness score for class y=0 is 0.

    """

    unique_classes = torch.sort(torch.unique(y))[0] # For example: [doctor, nurse, engineer]
    assert total_no_main_classes == 2 # only valid for two classes where the negative class is assumed to be y=0
    assert len(unique_classes) <= 2 # as there can't be more than 2 classes

    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0] # For example: [Male, Female]
    group_fairness = {} # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])

    positive_rate = torch.mean((preds == 1).float())  # prob(pred=doctor/y=doctor)

    '''
    it will have a structure of 
    {
        'doctor': {
            'm' : 0.5, 
            'f' : 0.6
            }, 
        'model': {
        'm': 0.5,
        'f': 0.7,
        }
    '''
    for uc in unique_classes: # iterating over each class say: uc=doctor for the first iteration
        group_fairness[uc] = {}

        for group in unique_groups: # iterating over each group say: group=male for the firt iteration
            mask_pos = (s == group)
            if uc == 1:
                 # find instances with y=doctor and s=male
                g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
                g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            else: # uc = 0 which in our case is negative class
                g_fairness_pos = torch.tensor(0.0).to(device) # TODO: check if g_fairness_pos in the obove if condition is of the same type
            fairness[mask_pos] = g_fairness_pos
            group_fairness[uc][group] = g_fairness_pos
            fairness_lookup[int(uc.item()),int(group.item())] = g_fairness_pos


    return group_fairness, fairness_lookup


def equal_opportunity(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """
    We assume y to be 0,1  that is y can only take 1 or 0 as the input.

    y = 1 is considered to be a positive class while y=0 is considered to be a negative class. This is
    unlike other methods where y=-1 is considered to be negative class.

    In demographic parity the fairness score for class y=0 is 0.

    """

    unique_classes = torch.sort(torch.unique(y))[0] # For example: [doctor, nurse, engineer]
    assert total_no_main_classes == 2 # only valid for two classes where the negative class is assumed to be y=0
    assert len(unique_classes) <= 2 # as there can't be more than 2 classes

    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0] # For example: [Male, Female]
    group_fairness = {} # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])

    positive_rate = torch.mean((preds[y == 1] == 1).float())  # prob(pred=doctor/y=doctor)

    '''
    it will have a structure of 
    {
        'doctor': {
            'm' : 0.5, 
            'f' : 0.6
            }, 
        'model': {
        'm': 0.5,
        'f': 0.7,
        }
    '''
    for uc in unique_classes: # iterating over each class say: uc=doctor for the first iteration
        group_fairness[uc] = {}

        for group in unique_groups: # iterating over each group say: group=male for the firt iteration
            mask_pos = torch.logical_and(y == uc, s == group)
            if uc == 1:
                 # find instances with y=doctor and s=male
                g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
                g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            else: # uc = 0 which in our case is negative class
                g_fairness_pos = torch.tensor(0.0).to(device) # TODO: check if g_fairness_pos in the obove if condition is of the same type
            fairness[mask_pos] = g_fairness_pos # imposing the scores on the mask.
            group_fairness[uc][group] = g_fairness_pos
            fairness_lookup[int(uc.item()),int(group.item())] = g_fairness_pos


    return group_fairness, fairness_lookup


def calculate_demographic_parity(preds,y,s, other_params):
    """assumes two classes with positive class as y=1"""
    device = other_params['device']
    total_no_aux_classes = other_params['total_no_aux_classes']
    total_no_main_classes = other_params['total_no_main_classes']
    group_fairness, fairness_lookup = demographic_parity(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0)
    scores = []
    for key, value in group_fairness.items():
        if int(key.item()) == 1:
            for key1, value1 in value.items():
                scores.append(value1.item())
    return scores, group_fairness


def calculate_equal_opportunity(preds,y,s, other_params):
    """assumes two classes with positive class as y=1"""
    device = other_params['device']
    total_no_aux_classes = other_params['total_no_aux_classes']
    total_no_main_classes = other_params['total_no_main_classes']
    group_fairness, fairness_lookup = equal_opportunity(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0)
    scores = []
    for key, value in group_fairness.items():
        if int(key.item()) == 1:
            for key1, value1 in value.items():
                scores.append(value1.item())
    return scores, group_fairness


def calculate_equal_odds(preds,y,s, other_params):
    """assumes two classes with positive class as y=1"""
    device = other_params['device']
    total_no_aux_classes = other_params['total_no_aux_classes']
    total_no_main_classes = other_params['total_no_main_classes']
    group_fairness, fairness_lookup = equal_odds(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0)
    sorted_class = np.sort([key.item() for key,value in group_fairness.items()])
    group_fairness = {key.item():value for key,value in group_fairness.items()}
    scores = []
    for key in sorted_class:
        for key1, value1 in group_fairness[key].items():
            scores.append(value1.item())
    return scores, group_fairness


def calculate_grms(preds, y, s, other_params=None):
    unique_classes = torch.sort(torch.unique(y))[0]  # For example: [doctor, nurse, engineer]
    unique_groups = torch.sort(torch.unique(s))[0]  # For example: [Male, Female]
    group_fairness = {}  # a dict which keeps a track on how fairness is changing

    '''
    it will have a structure of 
    {
        'doctor': {
            'm' : 0.5, 
            'f' : 0.6
            }, 
        'model': {
        'm': 0.5,
        'f': 0.7,
        }
    '''
    for uc in unique_classes:  # iterating over each class say: uc=doctor for the first iteration
        group_fairness[uc] = {}
        positive_rate = torch.mean((preds[y == uc] == uc).float())  # prob(pred=doctor/y=doctor)
        for group in unique_groups:  # iterating over each group say: group=male for the firt iteration
            mask_pos = torch.logical_and(y == uc, s == group)  # find instances with y=doctor and s=male
            g_fairness_pos = torch.mean((preds[mask_pos] == uc).float())
            temp = g_fairness_pos.item()
            if math.isnan(temp):
                temp = 0.0
            group_fairness[uc][group] = temp
        group_fairness[uc]['total_acc'] = positive_rate

    scores = []
    for key,value in group_fairness.items():
        temp = [value_1 for key_1, value_1 in value.items()]
        gender_1, gender_2 = temp[0], temp[1]
        scores.append((gender_1-gender_2)**2)

    return [np.sqrt(np.mean(scores))], group_fairness


def calculate_acc_diff(preds, y, s, other_params=None):
    # unique_classes = torch.sort(torch.unique(y))[0]  # For example: [doctor, nurse, engineer]
    all_acc= []
    main_acc = torch.mean((preds == y).float())
    all_acc.append(main_acc)

    unique_groups = torch.sort(torch.unique(s))[0]  # For example: [Male, Female]
    group_fairness = {}  # a dict which keeps a track on how fairness is changing

    '''
    it will have a structure of 
    {
        'doctor': {
            'm' : 0.5, 
            'f' : 0.6
            }, 
        'model': {
        'm': 0.5,
        'f': 0.7,
        }
    '''
    # for uc in unique_classes:  # iterating over each class say: uc=doctor for the first iteration
    #     group_fairness[uc] = {}
    #     positive_rate = torch.mean((preds[y == uc] == uc).float())  # prob(pred=doctor/y=doctor)
    for group in unique_groups:  # iterating over each group say: group=male for the firt iteration
        mask_pos = s == group  # find instances with y=doctor and s=male
        acc_class = torch.mean((preds[mask_pos] == y[mask_pos]).float())
        temp = acc_class.item()
        if math.isnan(temp):
            temp = 0.0
        all_acc.append(temp)

    scores = abs(all_acc[1] - all_acc[2])
    assert len(all_acc) == 3 # there are just two groups.

    return [scores], all_acc

def calculate_true_rates(preds, y, s, other_params):
    '''
    inspired from https://github.com/HanXudong/Diverse_Adversaries_for_Mitigating_Bias_in_Training/blob/b5b4c99ada17b3c19ab2ae8789bb56058cb72643/networks/eval_metrices.py#L14
    :param preds:
    :param y:
    :param s:
    :param other_params:
    :return:
    '''
    unique_group = torch.sort(torch.unique(s))[0]
    assert len(unique_group) == 2

    g1_preds = preds[s == 1]
    g1_labels = y[s == 1]

    g0_preds = preds[s == 0]
    g0_labels = y[s == 0]


    tn0, fp0, fn0, tp0 = confusion_matrix(g0_labels.cpu().detach().numpy(), g0_preds.cpu().detach().numpy()).ravel()
    TPR0 = tp0/(fn0+tp0)
    TNR0 = tn0/(fp0+tn0)

    tn1, fp1, fn1, tp1 = confusion_matrix(g1_labels.cpu().detach().numpy(), g1_preds.cpu().detach().numpy()).ravel()
    TPR1 = tp1/(fn1+tp1)
    TNR1 = tn1/(tn1+fp1)

    TPR_gap = TPR0-TPR1
    TNR_gap = TNR0-TNR1

    return [TPR_gap,TNR_gap], ((abs(TPR_gap)+abs(TNR_gap))/2.0)


def custom_equal_odds(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """

    :param preds: output/prediction of the model
    :param y: actual/ground/gold label
    :param s: aux output/ protected demographic attribute
    :param epsilon:
    :return:
    """

    unique_classes = torch.sort(torch.unique(y))[0] # For example: [doctor, nurse, engineer]
    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0] # For example: [Male, Female]
    group_fairness = {} # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])
    '''
    it will have a structure of 
    {
        'doctor': {
            'm' : 0.5, 
            'f' : 0.6
            }, 
        'model': {
        'm': 0.5,
        'f': 0.7,
        }
    '''
    for uc in unique_classes: # iterating over each class say: uc=doctor for the first iteration
        group_fairness[uc] = {}
        positive_rate = torch.mean((preds[y==uc] == uc).float()) # prob(pred=doctor/y=doctor)
        for group in unique_groups: # iterating over each group say: group=male for the firt iteration
            mask_pos = torch.logical_and(y==uc, s==group) # find instances with y=doctor and s=male
            g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
            g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            fairness[mask_pos] = g_fairness_pos
            group_fairness[uc][group] = g_fairness_pos
            fairness_lookup[int(uc.item()),int(group.item())] = g_fairness_pos


    return group_fairness, fairness_lookup


def calculate_ddp_dde(preds, y, s, other_params):
    preds = preds.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    s = s.detach().cpu().numpy()
    positive_rate_prot = get_positive_rate(preds[s == 0], y[s == 0])
    positive_rate_unprot = get_positive_rate(preds[s == 1], y[s == 1])
    true_positive_rate_prot = get_true_positive_rate(preds[s == 0], y[s == 0])
    true_positive_rate_unprot = get_true_positive_rate(preds[s == 1], y[s == 1])
    DDP = positive_rate_unprot - positive_rate_prot
    DEO = true_positive_rate_unprot - true_positive_rate_prot

    return [DDP, DEO], ((abs(DDP)+abs(DEO))/2.0)


def get_positive_rate(y_predicted, y_true):
    """Compute the positive rate for given predictions of the class label.
    Parameters
    ----------
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).
    y_true: numpy array
        The true class labels of shape=(number_points,).
    Returns
    ---------
    pr: float
        The positive rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    pr = (tp+fp) / (tp+fp+tn+fn)
    return pr

def get_true_positive_rate(y_predicted, y_true):
    """Compute the true positive rate for given predictions of the class label.
    Parameters
    ----------
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).
    y_true: numpy array
        The true class labels of shape=(number_points,).
    Returns
    ---------
    tpr: float
        The true positive rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    tpr = tp / (tp+fn)
    return tpr


def calculate_multiple_things(preds, y, s, other_params):
    """
    do something here. Find ones which are useful and bunch them together to form easy dict
    """
    grms_score, grms_group_fairness = calculate_grms(preds, y, s, other_params)
    acc_diff_scores, acc_diff_all_acc = calculate_acc_diff(preds, y, s, other_params)
    [ddp,deo], _ = calculate_ddp_dde(preds, y, s, other_params)
    [tpr_gap, tnr_gap], avg_tpr_tnr = calculate_true_rates(preds, y, s, other_params)

    scores = {
        'grms': grms_score,
        'acc_diff': acc_diff_scores,
        'ddp': ddp,
        'deo': deo,
        'tpr_gap': tpr_gap,
        'tnr_gap': tnr_gap,
        'avg_tpr_tnr': avg_tpr_tnr
    }

    return grms_score, scores


def calculate_multiple_things_blog(preds, y, s, other_params):
    """
    do something here. Find ones which are useful and bunch them together to form easy dict
    """
    grms_score, grms_group_fairness = calculate_grms(preds, y, s, other_params)
    acc_diff_scores, acc_diff_all_acc = calculate_acc_diff(preds, y, s, other_params)

    scores = {
        'grms': grms_score,
        'acc_diff': acc_diff_scores
    }

    return grms_score, scores


def calculate_dummy_fairness(preds, y, s, other_params):
    """does not calculate fairness. Just returns a dummy value. Useful in case where fairness need not be calculated."""
    return [0.0], {}

def get_fairness_function(fairness_function):
    if fairness_function.lower() == 'equal_odds':
        fairness_function = equal_odds
    elif fairness_function.lower() == 'demographic_parity':
        fairness_function = demographic_parity
    elif fairness_function.lower() == 'equal_opportunity':
        fairness_function = equal_opportunity
    else:
        print("following type are supported: equal_odds, demographic_parity, equal_opportunity")
        raise NotImplementedError
    return fairness_function

def get_fairness_score_function(fairness_score_function):
    # Fairness score function
    if fairness_score_function.lower() == 'grms':
        fairness_score_function = calculate_grms
    elif fairness_score_function.lower() == 'demographic_parity':
        fairness_score_function = calculate_demographic_parity
    elif fairness_score_function.lower() == 'equal_opportunity':
        fairness_score_function = calculate_equal_opportunity
    elif fairness_score_function.lower() == 'equal_odds':
        fairness_score_function = calculate_equal_odds
    elif fairness_score_function.lower() == 'true_rates':
        fairness_score_function = calculate_true_rates
    elif fairness_score_function.lower() == 'ddp_dde':
        fairness_score_function = calculate_ddp_dde
    elif fairness_score_function.lower() == 'acc_diff':
        fairness_score_function = calculate_acc_diff
    elif fairness_score_function.lower() == 'multiple_things':
        fairness_score_function = calculate_multiple_things
    elif fairness_score_function.lower() == 'calculate_multiple_things_blog':
        fairness_score_function = calculate_multiple_things_blog
    elif fairness_score_function.lower() == 'dummy_fairness':
        fairness_score_function = calculate_dummy_fairness
    else:
        print("following type are supported: grms, equal_odds, demographic_parity, equal_opportunity")
        raise NotImplementedError
    return fairness_score_function