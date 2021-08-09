# There are many fair-grad specific functionality which needs to be implemented.
# The structure is similar to simple_loop, but with slight changes throughout the codebase.


'''
    There is train and evaluate loop. And there is a thing which calls these loops with iterators.
'''
import math
import time
import torch
import logging
import numpy as np
from tqdm.auto import tqdm
from .common_functionality import *


logger = logging.getLogger(__name__)


def train(model, iterator, optimizer, criterion, device, accuracy_calculation_function, other_params):
    # still need to do something about the device thingy!!
    # here the iterator is a list. With first iterator being the training one and second iterator
    # being the one to calculate fairness. Do note that in default cases the second iterator which
    # is used to calculate fairness is same as first iterator.

    model.train()

    train_iterator, fairness_iterator = iterator['train_iterator'], iterator['fairness_iterator']


    is_gradient_reversal = other_params['gradient_reversal']
    is_adv = other_params['is_adv'] # adv
    is_regression = other_params['is_regression']
    task = other_params['task']
    fairness_lookup = other_params['fairness_lookup']
    fairness_function = other_params['fairness_function']


    if task == 'domain_adaptation': # legacy code. Would not be useful here. But keeping it for consistency.
        assert is_adv == True

    # tracking stuff
    epoch_loss_main = []
    epoch_loss_aux = []
    epoch_acc_aux = []
    epoch_acc_main = []
    epoch_total_loss = []

    all_hidden = []
    all_s = []

    fairness_all_aux, fairness_all_labels = [], []

    for items in tqdm(fairness_iterator):
        for key in items.keys(): # moving them to the correct device.
            items[key] = items[key].to(device)
        fairness_all_aux.append(items['aux'])  # aux label.
        fairness_all_labels.append(items['labels']) # main task label.

    fairness_all_aux = torch.cat(fairness_all_aux, out=torch.Tensor(len(fairness_all_aux), fairness_all_aux[0].shape[0])).to(device)
    fairness_all_labels = torch.cat(fairness_all_labels, out=torch.Tensor(len(fairness_all_labels), fairness_all_labels[0].shape[0])).to(device)
    total_no_aux_classes, total_no_main_classes = len(torch.unique(fairness_all_aux)), len(torch.unique(fairness_all_labels))
    fairness_all_preds = generate_predictions(model, fairness_iterator, device) # think of this as model.fit. Iterates over the iterator and returns model prediction.


    if not fairness_lookup.any():
        group_fairness, fairness_lookup = fairness_function(preds=fairness_all_preds, y=fairness_all_labels,
                                                            s=fairness_all_aux, device=device,
                                                            total_no_main_classes=total_no_main_classes,
                                                            total_no_aux_classes=total_no_aux_classes,
                                                            epsilon=0.0)
    else:
        group_fairness = other_params['group_fairness']

    if is_adv:
        loss_aux_scale = other_params['loss_aux_scale']

    for items in tqdm(train_iterator):

        for key in items.keys():
            items[key] = items[key].to(device)

        items['gradient_reversal'] = is_gradient_reversal
        optimizer.zero_grad()
        output = model(items)

        if task == 'domain_adaptation':
            output['prediction'] = output['prediction'][items['aux'] == 0]  # 0 is the src domain.
            items['labels'] = items['labels'][items['aux'] == 0]


        if is_regression:
            loss_main = criterion(output['prediction'].squeeze(), items['labels'].squeeze())
            if is_adv:
                loss_aux = criterion(output['adv_output'].squeeze(), items['aux'].squeeze())
        else:
            loss_main = criterion(output['prediction'], items['labels'])
            if is_adv:
                loss_aux = criterion(output['adv_output'], items['aux'])


        acc_main = accuracy_calculation_function(output['prediction'], items['labels'])
        if is_adv:
            acc_aux = accuracy_calculation_function(output['adv_output'], items['aux'])
            total_loss = loss_main + (loss_aux_scale * loss_aux)
        else:
            total_loss = loss_main
            loss_aux = torch.tensor(0.0)
            acc_aux = torch.tensor(0.0)

        all_hidden.append(output['hidden'].detach().cpu().numpy())
        all_s.append(items['aux'])

        fairness = fairness_lookup[items['labels'], items['aux']]

        if other_params['normalize_fairness']:
            normalization = torch.mean(1.0-fairness)
        else:
            normalization = torch.tensor(1.0)

        total_loss = torch.mean(total_loss * (1 - fairness.to(device)/normalization.to(device)))
        total_loss.backward()
        optimizer.step()

        # fair grad calculations
        fairness_all_preds = generate_predictions(model, fairness_iterator, device)
        interm_group_fairness, interm_fairness_lookup = fairness_function(preds=fairness_all_preds, y=fairness_all_labels,
                                                            s=fairness_all_aux, device=device,
                                                            total_no_main_classes=total_no_main_classes,
                                                            total_no_aux_classes=total_no_aux_classes,
                                                            epsilon=0.0)

        # Clipping
        fairness_lookup = fairness_lookup + interm_fairness_lookup
        if other_params['clip_fairness']:
            fairness_lookup = torch.clip(fairness_lookup, None, 1.0)



        epoch_loss_main.append(torch.mean(loss_main).item())
        epoch_acc_main.append(acc_main.item())
        epoch_loss_aux.append(torch.mean(loss_aux).item())
        epoch_acc_aux.append(acc_aux.item())
        epoch_total_loss.append(total_loss.item())

    return_output = {
        'epoch_total_loss': np.mean(epoch_total_loss),
        'epoch_loss_main': np.mean(epoch_loss_main),
        'epoch_acc_main': np.mean(epoch_acc_main),
        'epoch_loss_aux': np.mean(epoch_loss_aux),
        'epoch_acc_aux': np.mean(epoch_acc_aux)
    }

    other_data = {
        'all_hidden' : np.row_stack(all_hidden),
        'all_s' : torch.cat(all_s, out=torch.Tensor(len(all_s), all_s[0].shape[0])).to(device),
        'group_fairness': group_fairness,
        'fairness_lookup': fairness_lookup
    }

    return return_output, other_data

def evaluate(model, iterator, optimizer, criterion, device, accuracy_calculation_function, other_params):
    # works same as train loop with few exceptions. It does have code repetation
    model.eval()

    all_preds, all_hidden = [], []
    y,all_s = [], []

    # tracking stuff
    epoch_loss_main = []
    epoch_loss_aux = []
    epoch_acc_aux = []
    epoch_acc_main = []
    epoch_total_loss = []



    is_gradient_reversal = other_params['gradient_reversal']
    is_adv = other_params['is_adv'] # adv
    is_regression = other_params['is_regression']
    task = other_params['task']
    if task == 'domain_adaptation':
        assert is_adv == True

    if is_adv:
        loss_aux_scale = other_params['loss_aux_scale']

    with torch.no_grad():
        for items in tqdm(iterator):
            for key in items.keys():
                items[key] = items[key].to(device)
            items['gradient_reversal'] = is_gradient_reversal
            output = model(items)

            # if task == 'domain_adaptation':
            #     output['prediction'] = output['prediction'][items['aux'] == 1]  # 0 is the src domain.
            #     items['labels'] = items['labels'][items['aux'] == 0]

            if is_regression:
                loss_main = criterion(output['prediction'].squeeze(), items['labels'].squeeze())
                if is_adv:
                    loss_aux = criterion(output['adv_output'].squeeze(), items['aux'].squeeze())
            else:
                loss_main = criterion(output['prediction'], items['labels'])
                if is_adv:
                    loss_aux = criterion(output['adv_output'], items['aux'])

            all_preds.append(output['prediction'].argmax(1))
            all_hidden.append(output['hidden'].detach().cpu().numpy())
            y.append(items['labels'])

            if "aux" in items.keys():
                all_s.append(items['aux'])

            acc_main = accuracy_calculation_function(output['prediction'], items['labels'])
            if is_adv:
                acc_aux = accuracy_calculation_function(output['adv_output'], items['aux'])
                total_loss = loss_main + (loss_aux_scale * loss_aux)
            else:
                total_loss = loss_main
                loss_aux = torch.tensor(0.0)
                acc_aux = torch.tensor(0.0)


            epoch_loss_main.append(torch.mean(loss_main).item())
            epoch_acc_main.append(acc_main.item())
            epoch_loss_aux.append(torch.mean(loss_aux).item())
            epoch_acc_aux.append(acc_aux.item())
            epoch_total_loss.append(torch.mean(total_loss).item())

        # fairness stuff !!
        all_preds = torch.cat(all_preds, out=torch.Tensor(len(all_preds), all_preds[0].shape[0])).to(device)
        y = torch.cat(y, out=torch.Tensor(len(y), y[0].shape[0])).to(device)
        all_s = torch.cat(all_s, out=torch.Tensor(len(all_s), all_s[0].shape[0])).to(device)
        # all_hidden = torch.cat(all_hidden, out=torch.Tensor(len(all_hidden), all_hidden[0].shape[0]))
        all_hidden = np.row_stack(all_hidden)
        extra_info = {}
        if "adult_multigroup_sensr" in other_params['dataset_metadata']['dataset_name']:
            # a hack for passing more info specifically for adult multigroup sensr
            extra_info['s_concat'] = other_params['dataset_metadata']['s_concat']
        grms, group_fairness = calculate_fairness_stuff(all_preds,y, all_s, other_params['fairness_score_function'],
                                                        device, extra_info) # update this after everything we need is available here.



        return_output = {
            'epoch_total_loss': np.mean(epoch_total_loss),
            'epoch_loss_main': np.mean(epoch_loss_main),
            'epoch_acc_main': np.mean(epoch_acc_main),
            'epoch_loss_aux': np.mean(epoch_loss_aux),
            'epoch_acc_aux': np.mean(epoch_acc_aux),
            'grms': grms,
            'group_fairness': group_fairness,
        }

        other_data = {
            'all_hidden': all_hidden,
            'all_s': all_s,
            'all_preds': all_preds
        }


        return return_output, other_data



def training_loop( n_epochs:int,
        model,
        iterator,
        optimizer,
        criterion,
        device,
        model_save_name,
        accuracy_calculation_function,
        wandb,
        other_params):

    # for iterator in iterators: # Now this is for k-fold kind of setup # This will not be the correct way to do it.
        # Logging would get weird. This needs to be called from somewhere else.

    best_valid_loss = 1*float('inf')
    best_valid_acc = -1*float('inf')
    best_test_acc = -1*float('inf')
    test_acc_at_best_valid_acc = -1*float('inf')
    group_fairness, fairness_lookup = {}, torch.zeros([1, 1])

    save_model = other_params['save_model']
    original_eps = other_params['eps']
    eps_scale = other_params['eps_scale']
    current_best_grms = [math.inf]
    test_acc_at_best_grms = 0.0

    use_lr_schedule = other_params['use_lr_schedule']
    lr_scheduler = other_params['lr_scheduler']

    mode_of_adv_loss_scale = other_params['mode_of_loss_scale'] # this is for adv.
    original_loss_aux_scale = other_params['loss_aux_scale']
    current_adv_scale = 0.0

    save_wrt_loss = other_params['save_wrt_loss']
    total_epochs = n_epochs # this needs to be thought in more depth

    dataset = other_params["dataset"]

    other_params['group_fairness'], other_params['fairness_lookup'] = {}, torch.zeros([1, 1])
    training_iterator = {
        'train_iterator': iterator['train_iterator'],
        'fairness_iterator': iterator['fairness_iterator']
    }

    if other_params['is_adv']:
        other_params['gradient_reversal'] = True
    else:
        other_params['gradient_reversal'] = False

    # update adv loss with every epoch
    def get_current_adv_scale(epoch_number, last_scale):
        if mode_of_adv_loss_scale == 'constant':
            current_scale = original_loss_aux_scale
            return current_scale
        if mode_of_adv_loss_scale == 'linear':
            current_scale = last_scale + original_loss_aux_scale * 1.0 / total_epochs
            return current_scale
        if mode_of_adv_loss_scale == 'exp':
            p_i = epoch_number / total_epochs
            current_scale = float(original_loss_aux_scale * (2.0 / (1.0 + np.exp(-7 * p_i)) - 1.0))
            return current_scale

    for epoch in range(n_epochs):
        current_adv_scale = get_current_adv_scale(epoch_number=epoch,
                                          last_scale=current_adv_scale)
        other_params['loss_aux_scale'] = current_adv_scale

        start_time = time.monotonic()
        logging.info("start of block - simple loop")
        logger.info(f"current adv scale is {current_adv_scale}")


        train_output, train_other_data = train(model, training_iterator, optimizer, criterion, device,
                                          accuracy_calculation_function, other_params)
        other_params['group_fairness'], other_params['fairness_lookup'] = train_other_data['group_fairness'],\
                                                                          train_other_data['fairness_lookup']

        # model.eps = 1000


        val_output, val_other_data = evaluate(model, iterator['valid_iterator'], optimizer, criterion, device,
                             accuracy_calculation_function, other_params)
        test_output, test_other_data = evaluate(model, iterator['test_iterator'], optimizer, criterion, device,
                             accuracy_calculation_function, other_params)

        # now here we need to create a train and test set. all_hidden, all_s

        print(test_output['group_fairness'])

        if "amazon" in dataset:
            train_preds = np.vstack([val_other_data['all_hidden'][:int(0.60*len(val_other_data['all_hidden']))],
                                     test_other_data['all_hidden'][:int(0.60*len(test_other_data['all_hidden']))]])

            test_preds = np.vstack([val_other_data['all_hidden'][int(0.60 * len(val_other_data['all_hidden'])):],
                                     test_other_data['all_hidden'][int(0.60 * len(test_other_data['all_hidden'])):]])


            train_labels = np.hstack([val_other_data['all_s'][:int(0.60*len(val_other_data['all_s']))],
                                     test_other_data['all_s'][:int(0.60*len(test_other_data['all_s']))]])

            test_labels = np.hstack([val_other_data['all_s'][int(0.60 * len(val_other_data['all_s'])):],
                                     test_other_data['all_s'][int(0.60 * len(test_other_data['all_s'])):]])

            leakage = calculate_leakage(train_preds, train_labels, test_preds, test_labels, method='svm')
            print(f'\t Leakage is {leakage}')

        if "adult_multigroup_sensr" in dataset:
            # now this is a test iterator used for checking the consistency of preditction. Thus
            # saving the prediction is at most important
            save_consistency_preds = []
            for gr_iterator in other_params['dataset_metadata']['gender_race_consistent']:
                cons_output, cons_other_data = evaluate(model, gr_iterator, optimizer, criterion, device,
                                                      accuracy_calculation_function, other_params)
                save_consistency_preds.append(cons_other_data['all_preds'])

            _temp = torch.logical_not(torch.logical_xor(save_consistency_preds[1], save_consistency_preds[0]))
            _temp = torch.logical_not(torch.logical_xor(save_consistency_preds[2], _temp))
            _temp = torch.logical_not(torch.logical_xor(save_consistency_preds[3], _temp))

            # cons_acc = torch.logical_and(save_consistency_preds[0], torch.logical_and(save_consistency_preds[1], torch.logical_and(save_consistency_preds[2], save_consistency_preds[3])))
            cons_acc = torch.sum(_temp)*1.0/save_consistency_preds[0].shape[0]
            print(cons_acc)

        # model.eps = other_params['eps']

        logger.info(f"valid dict: {val_output}")
        logger.info(f"test dict: {test_output}")

        # calculate leakage - still need to be implemented.
        # @TODO: Complete the implementation of the leakge.
        #  Should be as SGD with non-linearity as well as svm support.

        logging.info("end of block - three_phase_adv_block")

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if use_lr_schedule:
            lr_scheduler.step(val_output['epoch_total_loss'])


        if val_output['epoch_total_loss'] < best_valid_loss:
            logger.info(f"model saved as: {model_save_name}")
            best_valid_loss = val_output['epoch_total_loss']
            if save_model and save_wrt_loss:
                torch.save(model.state_dict(), model_save_name)

        if val_output['epoch_acc_main'] > best_valid_acc:
            best_valid_acc = val_output['epoch_acc_main']
            test_acc_at_best_valid_acc = test_output['epoch_acc_main']
            if save_model and not save_wrt_loss:
                torch.save(model.state_dict(), model_save_name)

        if test_output['epoch_acc_main'] > best_test_acc:
            best_test_acc = test_output['epoch_acc_main']

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_output["epoch_total_loss"]:.3f} | Train Acc: {train_output["epoch_acc_main"]}%')
        print(f'\t Val. Loss: {val_output["epoch_total_loss"]:.3f} |  Val. Acc: {val_output["epoch_acc_main"]}%')
        print(f'\t Test Loss: {test_output["epoch_total_loss"]:.3f} |  Val. Acc: {test_output["epoch_acc_main"]}%')


        if wandb:
            wandb.log({'train_' + key:value for key, value in train_output.items()})
            wandb.log({'val_' + key:value for key, value in val_output.items()})
            wandb.log({'test_' + key:value for key, value in test_output.items()})

    return best_test_acc, best_valid_acc, test_acc_at_best_valid_acc