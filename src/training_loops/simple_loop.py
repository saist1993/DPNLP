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

    model.train()

    is_gradient_reversal = other_params['gradient_reversal']
    is_adv = other_params['is_adv'] # adv
    is_regression = other_params['is_regression']
    task = other_params['task']
    if task == 'domain_adaptation':
        assert is_adv == True

    # tracking stuff
    epoch_loss_main = []
    epoch_loss_aux = []
    epoch_acc_aux = []
    epoch_acc_main = []
    epoch_total_loss = []


    if is_adv:
        loss_aux_scale = other_params['loss_aux_scale']

    for items in tqdm(iterator):

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

        total_loss.backward()
        optimizer.step()

        epoch_loss_main.append(loss_main.item())
        epoch_acc_main.append(acc_main.item())
        epoch_loss_aux.append(loss_aux.item())
        epoch_acc_aux.append(acc_aux.item())
        epoch_total_loss.append(total_loss.item())

    return_output = {
        'epoch_total_loss': np.mean(epoch_total_loss),
        'epoch_loss_main': np.mean(epoch_loss_main),
        'epoch_acc_main': np.mean(epoch_acc_main),
        'epoch_loss_aux': np.mean(epoch_loss_aux),
        'epoch_acc_aux': np.mean(epoch_acc_aux)
    }

    return return_output

def evaluate(model, iterator, optimizer, criterion, device, accuracy_calculation_function, other_params):
    # works same as train loop with few exceptions. It does have code repetation
    model.eval()

    all_preds = []
    y,s = [], []

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
            y.append(items['labels'])

            if "aux" in items.keys():
                s.append(items['aux'])

            acc_main = accuracy_calculation_function(output['prediction'], items['labels'])
            if is_adv:
                acc_aux = accuracy_calculation_function(output['adv_output'], items['aux'])
                total_loss = loss_main + (loss_aux_scale * loss_aux)
            else:
                total_loss = loss_main
                loss_aux = torch.tensor(0.0)
                acc_aux = torch.tensor(0.0)


            epoch_loss_main.append(loss_main.item())
            epoch_acc_main.append(acc_main.item())
            epoch_loss_aux.append(loss_aux.item())
            epoch_acc_aux.append(acc_aux.item())
            epoch_total_loss.append(total_loss.item())

        # fairness stuff !!
        all_preds = torch.cat(all_preds, out=torch.Tensor(len(all_preds), all_preds[0].shape[0])).to(device)
        y = torch.cat(y, out=torch.Tensor(len(y), y[0].shape[0])).to(device)
        s = torch.cat(s, out=torch.Tensor(len(s), s[0].shape[0])).to(device)
        grms, group_fairness = calculate_fairness_stuff(all_preds, y, s, other_params['fairness_score_function'], device)

        return_output = {
            'epoch_total_loss': np.mean(epoch_total_loss),
            'epoch_loss_main': np.mean(epoch_loss_main),
            'epoch_acc_main': np.mean(epoch_acc_main),
            'epoch_loss_aux': np.mean(epoch_loss_aux),
            'epoch_acc_aux': np.mean(epoch_acc_aux),
            'grms': grms,
            'group_fairness': group_fairness
        }

        return return_output


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

    if other_params['is_adv']:
        other_params['gradient_reversal'] = True

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
        train_output = train(model, iterator['train_iterator'], optimizer, criterion, device,
                                          accuracy_calculation_function, other_params)


        val_output = evaluate(model, iterator['valid_iterator'], optimizer, criterion, device,
                             accuracy_calculation_function, other_params)
        test_output = evaluate(model, iterator['test_iterator'], optimizer, criterion, device,
                             accuracy_calculation_function, other_params)

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