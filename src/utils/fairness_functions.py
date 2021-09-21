import math
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def old_equal_odds(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
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


def old_demographic_parity(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """
    I thyink this is wrong. verify this Michael
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
            mask_pos = (s == group) # gender = male
            if uc == 1:
                 # find instances with y=doctor and s=male
                g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
                g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
                fairness[mask_pos] = g_fairness_pos
            else: # uc = 0 which in our case is negative class
                g_fairness_pos = torch.tensor(0.0).to(device) # TODO: check if g_fairness_pos in the obove if condition is of the same type
            # fairness[mask_pos] = g_fairness_pos # This might have been a subtle bug. As first the zero uc gets wriiten and over that one uc gets written. we only want uc==1 to be written
            group_fairness[uc][group] = g_fairness_pos
            fairness_lookup[int(uc.item()),int(group.item())] = g_fairness_pos


    return group_fairness, fairness_lookup


def demographic_parity(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """This is the new implementation of demographic parity.
     In this case we are following very close to the definition proposed in the FairGrad Paper. """

    unique_classes = torch.sort(torch.unique(y))[0] # For example: [doctor, nurse, engineer]
    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0] # For example: [Male, Female]
    group_fairness = {} # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])
    left_hand_matrix = torch.zeros([total_no_main_classes, total_no_aux_classes])
    per_group_accuracy = torch.zeros([total_no_main_classes, total_no_aux_classes])


    for uc in unique_classes: # iterating over each class say: uc=doctor for the first iteration
        group_fairness[uc.item()] = {}
        positive_rate = torch.mean((preds == uc).float())

        for group in unique_groups: # iterating over each group say: group=male for the firt iteration

            mask_pos = torch.logical_and(s == group, y==uc) # gender = male
            mask_group = (s==group)

            g_fairness_pos = torch.mean((preds[mask_group] == uc).float()) - positive_rate
            g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            fairness[mask_pos] = g_fairness_pos
            group_fairness[uc.item()][group.item()] = g_fairness_pos.item()
            fairness_lookup[int(uc.item()),int(group.item())] = g_fairness_pos
            left_hand_matrix[int(uc.item()),int(group.item())] = positive_rate
            per_group_accuracy[int(uc.item()), int(group.item())] = torch.mean((preds[mask_pos] == y[mask_pos]).float()).item()

    return group_fairness, fairness_lookup, left_hand_matrix, per_group_accuracy



def equal_odds(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """This is the new implementation of demographic parity.
         In this case we are following very close to the definition proposed in the FairGrad Paper. """
    unique_classes = torch.sort(torch.unique(y))[0]  # For example: [doctor, nurse, engineer]
    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0]  # For example: [Male, Female]
    group_fairness = {}  # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])
    left_hand_matrix = torch.zeros([total_no_main_classes, total_no_aux_classes])
    per_group_accuracy = torch.zeros([total_no_main_classes, total_no_aux_classes])

    for uc in unique_classes:  # iterating over each class say: uc=doctor for the first iteration
        group_fairness[uc.item()] = {}
        positive_rate = torch.mean((preds[y==uc] == uc).float())

        for group in unique_groups:  # iterating over each group say: group=male for the firt iteration

            mask_pos = torch.logical_and(s == group, y == uc)  # gender = male
            # mask_group = (s == group)

            g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
            g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            fairness[mask_pos] = g_fairness_pos
            group_fairness[uc.item()][group.item()] = g_fairness_pos.item()
            fairness_lookup[int(uc.item()), int(group.item())] = g_fairness_pos
            left_hand_matrix[int(uc.item()), int(group.item())] = positive_rate
            per_group_accuracy[int(uc.item()), int(group.item())] = torch.mean((preds[mask_pos] == y[mask_pos]).float()).item()

    return group_fairness, fairness_lookup, left_hand_matrix, per_group_accuracy


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
    left_hand_matrix = torch.zeros([total_no_main_classes, total_no_aux_classes])
    per_group_accuracy = torch.zeros([total_no_main_classes, total_no_aux_classes])

    # positive_rate = torch.mean((preds[y == 1] == 1).float())  # prob(pred=doctor/y=doctor)

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
        group_fairness[uc.item()] = {}
        positive_rate = torch.mean((preds[y == uc] == uc).float())

        for group in unique_groups: # iterating over each group say: group=male for the firt iteration
            mask_pos = torch.logical_and(y == uc, s == group)
            if uc == 1:
                 # find instances with y=doctor and s=male
                g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
                g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            else: # uc = 0 which in our case is negative class
                g_fairness_pos = torch.tensor(0.0).to(device) # TODO: check if g_fairness_pos in the obove if condition is of the same type
            fairness[mask_pos] = g_fairness_pos # imposing the scores on the mask.
            group_fairness[uc.item()][group.item()] = g_fairness_pos.item()
            fairness_lookup[int(uc.item()),int(group.item())] = g_fairness_pos
            left_hand_matrix[int(uc.item()), int(group.item())] = positive_rate
            per_group_accuracy[int(uc.item()), int(group.item())] = torch.mean((preds[mask_pos] == y[mask_pos]).float()).item()


    return group_fairness, fairness_lookup, left_hand_matrix, per_group_accuracy

def accuracy_parity(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    unique_classes = torch.sort(torch.unique(y))[0] # For example: [doctor, nurse, engineer]
    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0] # For example: [Male, Female]
    group_fairness = {} # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])
    positive_rate = torch.mean((preds==y).float())
    left_hand_matrix = torch.zeros([total_no_main_classes, total_no_aux_classes])
    per_group_accuracy = torch.zeros([total_no_main_classes, total_no_aux_classes])

    for uc in unique_classes:
        group_fairness[uc.item()] = {}

    for group in unique_groups:
        mask_group = (s == group)
        g_fairness_pos = torch.mean((preds[mask_group] == y[mask_group]).float()) - positive_rate
        g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
        fairness[mask_group] = g_fairness_pos # all males have same score irrespective of lable.
        for uc in unique_classes:
            mask_pos = torch.logical_and(s == group, y == uc)
            group_fairness[uc.item()][group.item()] = g_fairness_pos.item() # all lables have same score for given protected attribute
            fairness_lookup[int(uc.item()), int(group.item())] = g_fairness_pos
            left_hand_matrix[int(uc.item()), int(group.item())] = positive_rate
            per_group_accuracy[int(uc.item()), int(group.item())] = torch.mean((preds[mask_pos] == y[mask_pos]).float()).item()

    return group_fairness, fairness_lookup, left_hand_matrix, per_group_accuracy







def old_demographic_parity_multiclass(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """
    This has been generalized for multiclass setup. This looks more close to equal odds!
    We assume y to be 0,1  that is y can only take 1 or 0 as the input.

    y = 1 is considered to be a positive class while y=0 is considered to be a negative class. This is
    unlike other methods where y=-1 is considered to be negative class.

    In demographic parity the fairness score for class y=0 is 0.

    """

    unique_classes = torch.sort(torch.unique(y))[0] # For example: [doctor, nurse, engineer]
    # assert total_no_main_classes == 2 # only valid for two classes where the negative class is assumed to be y=0
    # assert len(unique_classes) <= 2 # as there can't be more than 2 classes

    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0] # For example: [Male, Female]
    group_fairness = {} # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])

      # prob(pred=doctor/y=doctor)

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
        positive_rate = torch.mean((preds == uc).float())

        for group in unique_groups: # iterating over each group say: group=male for the firt iteration
            mask_pos = (s == group)
            # mask2 = (y == uc)
            # mask_pos = torch.logical_and(mask1, mask2)
            # if uc == 1:
            #      # find instances with y=doctor and s=male
            g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
            g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            # else: # uc = 0 which in our case is negative class
            #     g_fairness_pos = torch.tensor(0.0).to(device) # TODO: check if g_fairness_pos in the obove if condition is of the same type
            fairness[mask_pos] = g_fairness_pos
            group_fairness[uc][group] = g_fairness_pos
            fairness_lookup[int(uc.item()),int(group.item())] = g_fairness_pos


    return group_fairness, fairness_lookup

def old_equal_odds_multiclass(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """
    This has been generalized for multiclass setup. This looks more close to equal odds!
    We assume y to be 0,1  that is y can only take 1 or 0 as the input.

    y = 1 is considered to be a positive class while y=0 is considered to be a negative class. This is
    unlike other methods where y=-1 is considered to be negative class.

    In demographic parity the fairness score for class y=0 is 0.

    """

    unique_classes = torch.sort(torch.unique(y))[0] # For example: [doctor, nurse, engineer]
    # assert total_no_main_classes == 2 # only valid for two classes where the negative class is assumed to be y=0
    # assert len(unique_classes) <= 2 # as there can't be more than 2 classes

    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0] # For example: [Male, Female]
    group_fairness = {} # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])

      # prob(pred=doctor/y=doctor)

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
        positive_rate = torch.mean((preds[y==uc] == uc).float())

        for group in unique_groups: # iterating over each group say: group=male for the firt iteration
            mask1 = (s == group)
            mask2 = (y == uc)
            mask_pos = torch.logical_and(mask1, mask2)
            # if uc == 1:
            #      # find instances with y=doctor and s=male
            g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
            g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            # else: # uc = 0 which in our case is negative class
            #     g_fairness_pos = torch.tensor(0.0).to(device) # TODO: check if g_fairness_pos in the obove if condition is of the same type
            fairness[mask_pos] = g_fairness_pos
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
            g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) # this is wrong
            '''
                It should be TP/TP + FN 
                
                where TP is gold = doctor and pred = doctor 
                    FN is gold = doctor but pred != doctor
                    g_fairness_pos = TP / (TP + FN) 
            '''
            TP = torch.sum(preds[mask_pos] == y[mask_pos]).item()# pred == doctor
            FN = torch.sum(preds[mask_pos] != y[mask_pos]).item()
            temp = TP*1.0/(TP+FN)
            # temp = g_fairness_pos.item()
            if math.isnan(temp):
                temp = 0.0
            group_fairness[uc][group] = temp
        group_fairness[uc]['total_acc'] = positive_rate

    scores = []
    for key,value in group_fairness.items():
        temp = [value_1 for key_1, value_1 in value.items()]
        gender_1, gender_2 = temp[0], temp[1]
        scores.append((gender_1-gender_2)**2)

    return [np.sqrt(np.mean(scores))], {}

def calculate_per_class_acc_difference(preds, y, s, other_params=None):
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


def calculate_ddp_dde_multiclass(preds, y, s, other_params):

    unique_classes = torch.sort(torch.unique(y))[0]
    preds = preds.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    s = s.detach().cpu().numpy()
    DDP, DEO = [], []
    for uc in unique_classes:
        mask_prot = np.logical_and(s==0.0, y==np.full_like(y,uc))
        mask_unprot = np.logical_and(s==1.0, y==np.full_like(y,uc))
        unique_mask_prot = np.unique(mask_prot)
        unique_mask_unprot = np.unique(mask_unprot)

        positive_rate_prot = get_positive_rate_multiclass(preds[mask_prot], y[mask_prot])
        positive_rate_unprot = get_positive_rate_multiclass(preds[mask_unprot], y[mask_unprot])

        true_positive_rate_prot = get_true_positive_rate_multiclass(preds[mask_prot], y[mask_prot])
        true_positive_rate_unprot = get_true_positive_rate_multiclass(preds[mask_unprot], y[mask_unprot])
        DDP.append(positive_rate_unprot - positive_rate_prot)
        DEO.append(true_positive_rate_unprot - true_positive_rate_prot)

    DDP = np.average(DDP)
    DEO = np.average(DEO)

    return [DDP, DEO], ((abs(DDP)+abs(DEO))/2.0)

def calculate_diff_demographic_parity(preds, y, s, other_params):
    [DDP, DEO], _ = calculate_ddp_dde(preds, y, s, other_params)
    return DDP, {'ddp': DDP}

def calculate_diff_equal_opportunity(preds, y, s, other_params):
    [DDP, DEO], _ = calculate_ddp_dde(preds, y, s, other_params)
    return DEO, {'deo': DEO}


def get_positive_rate_multiclass(y_predicted, y_true):
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
    unique_y = np.unique(y_true)

    new_y_true = []
    for s in y_true:
        if s == unique_y:
            new_y_true.append(1.0)
        else:
            new_y_true.append(0.0)

    new_y_predicted = []
    for s in y_predicted:
        if s == unique_y:
            new_y_predicted.append(1.0)
        else:
            new_y_predicted.append(0.0)

    # tn, fp, fn, tp = confusion_matrix(np.asarray(new_y_true), np.asarray(new_y_predicted)).ravel()
    x = confusion_matrix(np.asarray(new_y_true), np.asarray(new_y_predicted)).ravel()
    if len(x) == 4:
        tn, fp, fn, tp = x
    elif len(x) == 1:
        tp = x
        fp, fn, tn = 0.0,0.0,0.0
    else:
        raise IOError
    pr = (tp+fp) / (tp+fp+tn+fn)
    return pr


def get_true_positive_rate_multiclass(y_predicted, y_true):
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
    unique_y = np.unique(y_true)

    new_y_true = []
    for s in y_true:
        if s == unique_y:
            new_y_true.append(1.0)
        else:
            new_y_true.append(0.0)

    new_y_predicted = []
    for s in y_predicted:
        if s == unique_y:
            new_y_predicted.append(1.0)
        else:
            new_y_predicted.append(0.0)


    # tn, fp, fn, tp = confusion_matrix(np.asarray(new_y_true), np.asarray(new_y_predicted)).ravel()
    x = confusion_matrix(np.asarray(new_y_true), np.asarray(new_y_predicted)).ravel()
    if len(x) == 4:
        tn, fp, fn, tp = x
    elif len(x) == 1:
        tp = x
        fp, fn, tn = 0.0, 0.0, 0.0
    # pr = (tp + fp) / (tp + fp + tn + fn)

    tpr = tp / (tp+fn)
    return tpr
    # return torch.mean((y_predicted == y_true).float())


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
    # return torch.mean((y_predicted == y_true).float())


def get_true_negative_rate(y_predicted, y_true):
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
    tnr = tn / (tn+fp)
    return tnr
    # return torch.mean((y_predicted == y_true).float())

def get_false_negative_rate(y_predicted, y_true):
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
    fnr = fn / (fn+tp)
    return fnr
    # return torch.mean((y_predicted == y_true).float())

def get_false_positive_rate(y_predicted, y_true):
    """Compute the true positive rate for given predictions of the class label.
    Parameters
    ----------
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).
    y_true: numpy array
        The true class labels of shape=(number_points,).
    Returns
    Returns
    ---------
    tpr: float
        The true positive rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    fpr = fp / (fp+tn)
    return fpr
    # return torch.mean((y_predicted == y_true).float())

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
    grms_score, grms_group_fairness = calculate_per_class_acc_difference(preds, y, s, other_params)
    acc_diff_scores, acc_diff_all_acc = calculate_acc_diff(preds, y, s, other_params)

    scores = {
        'grms': grms_score,
        'acc_diff': acc_diff_scores
    }

    return grms_score, scores


def calculate_dummy_fairness(preds, y, s, other_params):
    """does not calculate fairness. Just returns a dummy value. Useful in case where fairness need not be calculated."""
    return [0.0], {}


def calculate_fairness_adult_sensr_org(preds, y, s, other_params):
    """
    This is a very specifc fairness function which to be used just in the context of
    adult multi group sensr.
    :param preds:
    :param y:
    :param s:
    :param other_params:
    :return:
    """
    # hardcoding it for now -> automate this at some point of time in the future.
    gender_zero = [0,1]
    gender_one = [2,3]
    race_zero = [0,2]
    race_one = [1,3]

    gender_zero_index = np.logical_or(s==gender_zero[0] , s==gender_zero[1])
    gender_one_index = np.logical_or(s==gender_one[0] , s==gender_one[1])
    race_zero_index = np.logical_or(s==race_zero[0] , s==race_zero[1])
    race_one_index = np.logical_or(s==race_one[0] , s==race_one[1])

    y_zero_index = y==0
    y_one_index = y==1

    tpr_gender_zero_zero = get_true_positive_rate(preds[np.logical_and(gender_zero_index, y_zero_index)], y[np.logical_and(gender_zero_index, y_zero_index)])
    # tpr_gender_zero = get_true_positive_rate(preds[gender_zero_index], y[gender_zero_index])
    # tpr_gender_one = get_true_positive_rate(preds[gender_one_index], y[gender_one_index])
    grms_gender = 1
    
    tpr_gender_zero_zero = get_true_negative_rate(preds[np.logical_and(gender_zero_index, y_zero_index)], y[np.logical_and(gender_zero_index, y_zero_index)]) # zero on gender and zero on y
    tpr_gender_one_zero = get_true_negative_rate(preds[np.logical_and(gender_one_index, y_zero_index)], y[np.logical_and(gender_one_index, y_zero_index)]) # one on gender and zero on y
    
    tpr_gender_zero_one = get_true_positive_rate(preds[np.logical_and(gender_zero_index, y_one_index)], y[np.logical_and(gender_zero_index, y_one_index)]) # zero on gender and zero on y
    tpr_gender_one_one = get_true_positive_rate(preds[np.logical_and(gender_one_index, y_one_index)], y[np.logical_and(gender_one_index, y_one_index)]) # one on gender and zero on y

    tpr_race_zero_zero = get_true_negative_rate(preds[np.logical_and(race_zero_index, y_zero_index)], y[
        np.logical_and(race_zero_index, y_zero_index)])  # zero on gender and zero on y
    tpr_race_one_zero = get_true_negative_rate(preds[np.logical_and(race_one_index, y_zero_index)], y[
        np.logical_and(race_one_index, y_zero_index)])  # one on gender and zero on y

    tpr_race_zero_one = get_true_positive_rate(preds[np.logical_and(race_zero_index, y_one_index)], y[
        np.logical_and(race_zero_index, y_one_index)])  # zero on gender and zero on y
    tpr_race_one_one = get_true_positive_rate(preds[np.logical_and(race_one_index, y_one_index)], y[
        np.logical_and(race_one_index, y_one_index)])  # one on gender and zero on y

    gap_gender_zero = abs(tpr_gender_zero_zero - tpr_gender_one_zero)
    gap_gender_one = abs(tpr_gender_zero_one - tpr_gender_one_one)
    gender_rms = math.sqrt(((gap_gender_zero**2 + gap_gender_one**2)/2.0))


    gap_race_zero = abs(tpr_race_zero_zero - tpr_race_one_zero)
    gap_race_one = abs(tpr_race_zero_one - tpr_race_one_one)
    race_rms = math.sqrt(((gap_race_zero**2 + gap_race_one**2)/2.0))

    fairness = {
        'gender_rms': gender_rms,
        'race_rms': race_rms,
        'max_gap_gender': max(gap_gender_one, gap_gender_zero),
        'max_gap_race': max(gap_race_one,gap_race_zero)
    }

    return gender_rms, fairness


def calculate_fairness_adult_sensr(preds, y, s, other_params):
    """
    This is a very specifc fairness function which to be used just in the context of
    adult multi group sensr.
    :param preds:
    :param y:
    :param s:
    :param other_params:
    :return:
    """
    # hardcoding it for now -> automate this at some point of time in the future.
    gender_zero = [0, 1]
    gender_one = [2, 3]
    race_zero = [0, 2]
    race_one = [1, 3]
    favourable_label = 1.0
    unfavourable_label = 0.0

    preds = preds.detach().cpu().numpy()
    s = s.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y = y.ravel()
    preds = preds.ravel()

    y_true_pos = (y == favourable_label)  # True for all instances with y=1
    y_true_neg = (y == unfavourable_label)  # True for all instances with y=0

    def temp(cond_vector):
        # all instances with r=1 (favourable condition) and pred=1
        y_pred_pos = np.logical_and(preds == favourable_label, cond_vector)

        # all instances with r=1 (favourable condition) and pred=0
        y_pred_neg = np.logical_and(preds == unfavourable_label, cond_vector)

        return dict(
            TP = np.sum(np.logical_and(y_true_pos, y_pred_pos), dtype=np.float64),
            FP = np.sum(np.logical_and(y_true_neg, y_pred_pos), dtype=np.float64),
            TN = np.sum(np.logical_and(y_true_neg, y_pred_neg), dtype=np.float64),
            FN = np.sum(np.logical_and(y_true_pos, y_pred_neg), dtype=np.float64),
            positive = np.sum(np.logical_and(y_true_pos, cond_vector), dtype=np.float64),
            negative =np.sum(np.logical_and(y_true_neg, cond_vector), dtype=np.float64)
        )
    # gender
    # find all gender 1
    cond_vec = np.logical_or(s==gender_one[0] , s==gender_one[1])
    gender_1_rates = temp(cond_vec)

    # find all gender 0
    cond_vec = np.logical_or(s==gender_zero[0] , s==gender_zero[1])
    gender_0_rates = temp(cond_vec)


    gap_gender_positive = abs((gender_0_rates['FP']/gender_0_rates['negative']) - \
                          (gender_1_rates['FP']/gender_1_rates['negative']))

    gap_gender_negative = abs((gender_0_rates['FN'] / gender_0_rates['positive']) - (
                            gender_1_rates['FN'] / gender_1_rates['positive']))

    gender_rms = math.sqrt(((gap_gender_positive**2 + gap_gender_negative**2)/2.0))


    ### now for race. Same stuff

    cond_vec = np.logical_or(s == race_one[0], s == race_one[1])
    race_1_rates = temp(cond_vec)

    # find all gender 0
    cond_vec = np.logical_or(s == race_zero[0], s == race_zero[1])
    race_0_rates = temp(cond_vec)

    gap_race_positive = abs((race_0_rates['FP'] / race_0_rates['negative']) - \
                              (race_1_rates['FP'] / race_1_rates['negative']))

    gap_race_negative = abs((race_0_rates['FN'] / race_0_rates['positive']) - (
            race_1_rates['FN'] / race_1_rates['positive']))

    race_rms = math.sqrt(((gap_race_positive ** 2 + gap_race_negative ** 2) / 2.0))


    fairness = {
        'gender_rms': gender_rms,
        'race_rms': race_rms,
        'max_gap_gender': max(gap_gender_positive, gap_gender_negative),
        'max_gap_race': max(gap_race_positive, gap_race_negative)
    }

    return gender_rms, fairness


def calculate_fairness_demographic_parity_fairgrad(preds, y, s, other_params):
    """
        In other_parmas group_fairness info is imperative
        group_fairness = [[0.1,0.2],[0.3,0.4],[0.1,0.1]]
    """




def get_fairness_function(fairness_function):
    if fairness_function.lower() == 'equal_odds':
        fairness_function = equal_odds
    elif fairness_function.lower() == 'demographic_parity':
        fairness_function = demographic_parity
    elif fairness_function.lower() == 'equal_opportunity':
        fairness_function = equal_opportunity
    elif fairness_function.lower() == 'accuracy_parity':
        fairness_function = accuracy_parity
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
    elif fairness_score_function.lower() == 'fairness_adult_sensr':
        fairness_score_function = calculate_fairness_adult_sensr
    elif fairness_score_function.lower() == 'diff_demographic_parity':
        fairness_score_function = calculate_diff_demographic_parity
    elif fairness_score_function.lower() == 'diff_equal_opportunity':
        fairness_score_function = calculate_diff_equal_opportunity
    elif fairness_score_function.lower() == 'calculate_ddp_dde_multiclass':
        fairness_score_function = calculate_ddp_dde_multiclass
    else:
        print("following type are supported: grms, equal_odds, demographic_parity, equal_opportunity")
        raise NotImplementedError
    return fairness_score_function