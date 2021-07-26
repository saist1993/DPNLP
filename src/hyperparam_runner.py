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
    parser.add_argument('--noise_layer', '-noise_layer', help="True for using noise; else false (default).", type=str)
    # parser.add_argument('--eps', '-eps', help="[0.1, 0.2, 2.0, 2.4]", type=str)

    parser.add_argument('--eps_list', '-eps_list', nargs="*", help="--eps_list 0.1, 0.2, 2.0, 2.4", type=float)


    parser.add_argument('--is_adv', '-is_adv', help="True for using adv; else false (default).", type=str)
    parser.add_argument('--adv_start', '-adv_s', help="start of adv scale for increment increase for ex: 0.1", type=float)
    parser.add_argument('--adv_end', '-adv_e', help="end of adv scale 1.0", type=float)
    parser.add_argument('--epochs', '-epochs', help="epochs!", type=int)

    args = parser.parse_args()

    assert args.log_name != None
    assert args.dataset_name != None
    assert args.epochs != None

    # create main logging dir
    logs_dir = Path('../logs')
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
    noise_layer = str2bool(args.noise_layer)
    is_adv = str2bool(args.is_adv)
    bs = 64
    only_perturbate = True
    mode_of_loss_scale = 'linear' # linear atleast for amazon!
    # optimizer = 'sgd'


    lrs = [('adam', 0.001)]

    if noise_layer:
        # epss = map(float, args.eps.strip('[]').split(','))
        epss = args.eps_list
    else:
        epss = [0.0]

    if is_adv:
        adv_scales = [round(i,2) for i in np.arange(args.adv_start,args.adv_end,0.1)]
    else:
        adv_scales = [0.0]



    if dataset_name in ['blog', 'blog_v2']:
        fairness_score_function = 'calculate_multiple_things_blog'
    elif "amazon" in dataset_name:
        fairness_score_function = 'dummy_fairness'
    else:
        fairness_score_function = 'multiple_things'

    for eps in epss:
        for adv_scale in adv_scales:
            for optimizer, lr in lrs:
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
                         is_adv=is_adv,
                         adv_loss_scale=adv_scale,
                         use_pretrained_emb=False,
                         default_emb_dim=300,
                         save_test_pred=False,
                         noise_layer=noise_layer,
                         eps=eps,
                         is_post_hoc=False,
                         train_main_model=True,
                         use_wandb=False,
                         config_dict="",
                         experiment_name="hyper-param-search",
                         only_perturbate=only_perturbate,
                         mode_of_loss_scale=mode_of_loss_scale,
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
                         fair_grad=False,
                         reset_fairness=False,
                         use_adv_dataset=True,
                         use_lr_schedule=True,
                         fairness_function='demographic_parity',
                         fairness_score_function=fairness_score_function,
                         sample_specific_class=True
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

