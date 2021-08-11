import uuid
import logging
import argparse
import traceback
import numpy as np
from main import main
from pathlib import Path


create_dir = lambda dir_location: Path(dir_location).mkdir(parents=True, exist_ok=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', '-name', help="log file name", type=str)
    parser.add_argument('--dataset_name', '-dataset', help="name of the dataset", type=str)
    parser.add_argument('--epochs', '-epochs', help="epochs!", type=int)
    parser.add_argument('--use_lr_schedule', '-use_lr_schedule', help="use_lr_schedule!", type=str)
    parser.add_argument('--fairness_iterator', '-fairness_iterator', nargs="*", help="--fairness_iterator custom_1 custom_2 custom_3 train", type=str)
    # parser.add_argument('--use_clipping', '-use_clipping', help="employs clipping!", type=str)
    # parser.add_argument('--use_normalization', '-use_normalization', help="employs normalization", type=str)
    # parser.add_argument('--fairness_iterator', '-fairness_iterator', help="the type of fairness iterator to use", type=str)
    # parser.add_argument('--fairness_function', '-fairness_function', help="demographic_parity/equal_opportunity/equal_odds", type=str)

    args = parser.parse_args()

    assert args.log_name != None
    assert args.dataset_name != None
    assert args.epochs != None

    bs = 1000

    # create main logging dir
    logs_dir = Path('../logs/fair_grad')
    create_dir(logs_dir)

    # create dataset dir in logs_dir
    dataset_dir = logs_dir / Path(args.dataset_name)
    create_dir(dataset_dir)
    log_file_name = str(dataset_dir / Path(args.log_name)) + '.log'


    # logger init
    logging.basicConfig(filename=log_file_name,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logging.info(f"logging for {log_file_name}")
    logger = logging.getLogger('main')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    # setting up the run
    epochs = args.epochs
    dataset_name = args.dataset_name
    use_lr_schedule = str2bool(args.use_lr_schedule)
    # use_clipping = str2bool(args.use_clipping)
    # use_normalization =  str2bool(args.use_normalization)
    # fairness_iterator = args.fairness_iterator
    # fairness_function = args.fairness_function
    # fairness_score_function = fairness_function

    lrs = [('adam', 0.001)]

    # fairness_clippings = [True, False]
    # fairness_normalizations = [True, False]
    # fairness_functions = ['demographic_parity', 'equal_opportunity']
    # fairness_iterators = ['train', 'custom_1']

    fairness_clippings = [True]
    fairness_normalizations = [True]
    fairness_functions = ['demographic_parity']
    # fairness_iterators = ['custom_1', 'custom_2', 'custom_3']
    fairness_iterators = args.fairness_iterator


    for optimizer, lr in lrs:
        for fair_clip in fairness_clippings:
            for fair_norm in fairness_normalizations:
                for fairness_function in fairness_functions:
                    for fairness_iterator in fairness_iterators:
                        unique_id = str(uuid.uuid4())
                        try:
                            logger.info(f"start of run - {unique_id}")
                            main(emb_dim=300,
                                 spacy_model="en_core_web_sm",
                                 seed=1234,
                                 dataset_name=dataset_name,
                                 batch_size=bs,
                                 pad_token='<pad>',
                                 unk_token='<unk>',
                                 pre_trained_embeddings='../../bias-in-nlp/different_embeddings/simple_glove_vectors.vec',
                                 model_save_name='bilstm.pt',
                                 model='linear_adv',
                                 regression=False,
                                 tokenizer_type='simple',
                                 use_clean_text=True,
                                 max_length=None,
                                 epochs=epochs,
                                 learnable_embeddings=False,
                                 vocab_location=False,
                                 is_adv=False,
                                 adv_loss_scale=0.0,
                                 use_pretrained_emb=False,
                                 default_emb_dim=300,
                                 save_test_pred=False,
                                 noise_layer=False,
                                 eps=0.0,
                                 is_post_hoc=False,
                                 train_main_model=True,
                                 use_wandb=False,
                                 config_dict="",
                                 experiment_name="hyper-param-search",
                                 only_perturbate=False,
                                 mode_of_loss_scale='linear',
                                 training_loop_type='three_phase_custom',
                                 hidden_loss=False,
                                 hidden_l1_scale=0.5,
                                 hidden_l2_scale=0.5,
                                 reset_classifier=False,
                                 reset_adv=True,
                                 encoder_learning_rate_second_phase=0.01,
                                 classifier_learning_rate_second_phase=0.01,
                                 trim_data=True,
                                 eps_scale='constant',
                                 optimizer=optimizer,
                                 lr=lr,
                                 fair_grad=True,
                                 reset_fairness=False,
                                 use_adv_dataset=True,
                                 use_lr_schedule=use_lr_schedule,
                                 fairness_function=fairness_function,
                                 fairness_score_function="diff_" + fairness_function,
                                 sample_specific_class=True,
                                 calculate_leakage=False,
                                 clip_fairness=fair_clip,
                                 normalize_fairness=fair_norm,
                                 fairness_iterator=fairness_iterator
                                 )
                            logger.info(f"end of run - {unique_id}")
                        except KeyboardInterrupt:
                            raise IOError
                        except Exception:
                            error = traceback.print_exc()
                            print(error)
                            logger.info(error)
                            logger.info(f"run failed for some reason - {unique_id}")
                            logger.info(f"end of run - {unique_id}")
                            continue








