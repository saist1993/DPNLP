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
    parser.add_argument('--bs', '-bs', help="epochs!", type=int)
    parser.add_argument('--lr', '-lr', help="lr!", type=float)
    parser.add_argument('--use_lr_schedule', '-use_lr_schedule', help="use_lr_schedule!", type=str)
    parser.add_argument('--fairness_iterator', '-fairness_iterator', nargs="*", help="--fairness_iterator custom_1 custom_2 custom_3 train", type=str)
    # parser.add_argument('--seed', '-seed', help="1234", type=int)

    parser.add_argument('--seed', '-seed', nargs="*", help="--seed 2 4 8 16 32 64 128 256 512 42", type=int)
    parser.add_argument('--fairness_functions', '-fairness_functions', nargs="*", help="--fairness_functions accuracy_parity demographic_parity equal_odds equal_opportunity", type=str)
    parser.add_argument('--use_clipping', '-use_clipping', help="employs clipping!", type=str)
    parser.add_argument('--use_normalization', '-use_normalization', help="employs normalization", type=str)

    parser.add_argument('--simple_baseline', '-simple_baseline', help="employs simple baseline", type=str)

    parser.add_argument('--is_adv', '-is_adv', help="True for using adv; else false (default).", type=str)
    parser.add_argument('--adv_start', '-adv_s', help="start of adv scale for increment increase for ex: 0.1", type=float)
    parser.add_argument('--adv_end', '-adv_e', help="end of adv scale 1.0", type=float)
    # parser.add_argument('--fairness_iterator', '-fairness_iterator', help="the type of fairness iterator to use", type=str)
    # parser.add_argument('--fairness_function', '-fairness_function', help="demographic_parity/equal_opportunity/equal_odds", type=str)

    args = parser.parse_args()

    assert args.log_name != None
    assert args.dataset_name != None
    assert args.epochs != None

    if args.bs:
        bs = args.bs
    else:
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
                        filemode='w',
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
    is_simple_baseline = str2bool(args.simple_baseline)
    epochs = args.epochs
    dataset_name = args.dataset_name
    use_lr_schedule = str2bool(args.use_lr_schedule)
    is_adv = str2bool(args.is_adv)


    if args.use_clipping:
        fairness_clippings = [str2bool(args.use_clipping)]
    else:
        fairness_clippings = [True, False]

    if is_simple_baseline:
        fairness_clippings = [False]

    if args.use_normalization:
        fairness_normalizations =  [str2bool(args.use_normalization)]
    else:
        fairness_normalizations = [True, False]

    if is_simple_baseline:
        fairness_normalizations = [False]
    # fairness_iterator = args.fairness_iterator
    # fairness_function = args.fairness_function
    # fairness_score_function = fairness_function
    # fairness_functions = ['accuracy_parity']
    if args.fairness_functions:
        fairness_functions = args.fairness_functions
    else:
        fairness_functions = ['accuracy_parity', 'demographic_parity', 'equal_odds', 'equal_opportunity']
    # fairness_functions = ['demographic_parity']'


    if args.lr:
        lrs = [('sgd', args.lr)]
    else:
        lrs = [('sgd', 0.001)]
    fairness_iterators = args.fairness_iterator
    seeds = args.seed

    fair_grad = True

    if is_simple_baseline:
        fair_grad = False



    if is_adv:
        if args.adv_end == 0.0:
            adv_scales = [args.adv_start]
        else:
            adv_scales = [round(i,2) for i in np.arange(args.adv_start,args.adv_end,0.1)]
        fair_grad = False
    else:
        adv_scales = [0.0]


    if fair_grad: #### A hack
        hidden_l1_scale = 1.0
    else:
        hidden_l1_scale = 0.0

    fair_grad = True #### A hack




    for adv_scale in adv_scales:
        for seed in seeds:
            for optimizer, lr in lrs:
                for fair_clip in fairness_clippings:
                    for fair_norm in fairness_normalizations:
                        for fairness_function in fairness_functions:
                            for fairness_iterator in fairness_iterators:
                                unique_id = str(uuid.uuid4())
                                epochs = args.epochs
                                try:
                                    logger.info(f"start of run - {unique_id}")
                                    main(emb_dim=300,
                                         spacy_model="en_core_web_sm",
                                         seed=seed,
                                         dataset_name=dataset_name,
                                         batch_size=bs,
                                         pad_token='<pad>',
                                         unk_token='<unk>',
                                         pre_trained_embeddings='../../bias-in-nlp/different_embeddings/simple_glove_vectors.vec',
                                         model_save_name='bilstm.pt',
                                         model='linear_adv', # linear_adv
                                         regression=False,
                                         tokenizer_type='simple',
                                         use_clean_text=True,
                                         max_length=None,
                                         epochs=epochs,
                                         learnable_embeddings=False,
                                         vocab_location=False,
                                         is_adv=is_adv,
                                         adv_loss_scale=adv_scale,
                                         use_pretrained_emb=False,
                                         default_emb_dim=300,
                                         save_test_pred=False,
                                         noise_layer=False,
                                         eps=0.0,
                                         is_post_hoc=False,
                                         train_main_model=True,
                                         use_wandb=False,
                                         config_dict="simple",
                                         experiment_name="hyper-param-search",
                                         only_perturbate=False,
                                         mode_of_loss_scale='exp',
                                         training_loop_type='three_phase_custom',
                                         hidden_loss=False,
                                         hidden_l1_scale=hidden_l1_scale,
                                         hidden_l2_scale=0.5,
                                         reset_classifier=False,
                                         reset_adv=True,
                                         encoder_learning_rate_second_phase=0.01,
                                         classifier_learning_rate_second_phase=0.01,
                                         trim_data=False,
                                         eps_scale='constant',
                                         optimizer=optimizer,
                                         lr=lr,
                                         fair_grad=fair_grad,
                                         reset_fairness=False,
                                         use_adv_dataset=True,
                                         use_lr_schedule=use_lr_schedule,
                                         fairness_function=fairness_function,
                                         fairness_score_function="dummy_fairness",
                                         sample_specific_class=True,
                                         calculate_leakage=False,
                                         clip_fairness=fair_clip,
                                         normalize_fairness=fair_norm,
                                         fairness_iterator=fairness_iterator,
                                         supervised_da=False,
                                         apply_noise_to_adv=True
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








