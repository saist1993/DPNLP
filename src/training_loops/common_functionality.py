import torch


def calculate_fairness_stuff(all_preds, y, s, fairness_score_function, device):

    total_no_main_classes, total_no_aux_classes = len(torch.unique(y)), len(torch.unique(s))
    scoring_function_params = {
        'device': device,
        'total_no_aux_classes': total_no_aux_classes,
        'total_no_main_classes': total_no_main_classes
    }
    grms, group_fairness = fairness_score_function(all_preds, y, s, scoring_function_params)

    return grms, group_fairness

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    import numpy as np

    mode_of_adv_loss_scale = 'exp'
    n_epochs = 10
    original_loss_aux_scale = 10.0
    total_epochs = n_epochs
    current_scale = 0.0

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

    for i in range(n_epochs):
        current_scale = get_current_adv_scale(i+1, current_scale)
        print(i, current_scale)