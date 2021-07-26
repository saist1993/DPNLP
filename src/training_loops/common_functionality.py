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
