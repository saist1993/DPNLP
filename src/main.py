# File where everything begins.

# Library Imports
import copy
import click
import random
import pickle
import gensim
import logging
import numpy as np
from functools import partial
from mytorch.utils.goodies import *
from typing import Optional, Callable

# Custom Imports
import config
from utils.misc import *
from models.linear import *
from training_loops.simple_loop import training_loop as simple_training_loop
from training_loops.diverse_adv_loop import training_loop as diverse_training_loop
from training_loops.simple_loop import k_fold_training_loop as k_fold_training_loop
from training_loops.fair_grad import training_loop as fair_grad_training_loop
from utils.fairness_functions import *
from tokenizer_wrapper import init_tokenizer
from create_data import generate_data_iterators

# Setting up logger
logger = logging.getLogger(__name__)

# Bunch of these arguments have no use right now. It will be useful once all the functionalities are implemented.
def main(emb_dim:int,
         spacy_model:str,
         seed:int,
         dataset_name:str,
         batch_size:int,
         pad_token:str,
         unk_token:str,
         pre_trained_embeddings:str,
         model_save_name:str,
         model:str,
         regression:bool,
         tokenizer_type:str,
         use_clean_text:bool,
         max_length:Optional[int],
         epochs:int,
         learnable_embeddings:bool,
         vocab_location:Optional[None],
         is_adv:bool,
         adv_loss_scale:float,
         use_pretrained_emb:bool,
         default_emb_dim:int,
         save_test_pred:bool,
         noise_layer:bool,
         eps:float,
         is_post_hoc:bool,
         train_main_model:bool,
         use_wandb:bool,
         config_dict:str,
         experiment_name:str,
         only_perturbate:bool,
         mode_of_loss_scale:str,
         training_loop_type:str,
         hidden_loss:bool,
         hidden_l1_scale:int,
         hidden_l2_scale:int,
         reset_classifier:bool,
         reset_adv:bool,
         encoder_learning_rate_second_phase:float,
         classifier_learning_rate_second_phase:float,
         trim_data:bool,
         eps_scale:str,
         optimizer:str,
         lr:float,
         fair_grad:bool,
         reset_fairness:bool,
         use_adv_dataset:bool,
         use_lr_schedule:bool,
         fairness_function:str,
         fairness_score_function:str,
         sample_specific_class:bool,
         calculate_leakage:bool,
         clip_fairness:bool,
         normalize_fairness:bool,
         fairness_iterator:str,
         supervised_da:bool,
         apply_noise_to_adv:bool,
         diverse_adversary:bool,
         diverse_adv_lambda:float
         ):
    '''
        A place keep all the design choices.

        -> For now not keeping use_adv_data. All data are assumed to be adversarial, up-till and unless they are not.
        Here adversarial refers to the protected/private attributes present in the data.

        -> For now these are the following task
            a) 'domain_adaptation'
            b) 'fairness/privacy'
            -> Although there is
    '''

    # logging all arguments
    logger.info(f"arguemnts: {locals()}")

    # wandb logging stuff
    if use_wandb:
        import wandb
        wandb.init(project='bias_in_nlp', entity='magnet', config=click.get_current_context().params)
        run_id = wandb.run.id
        logger.info(f"wandb run id is {run_id}")
    else:
        wandb = False

    # kind of model to use.
    if "amazon" in dataset_name:
        logger.info(f"model chossen amazon model")
        model_arch_params = config.amazon_model
    elif "adult_multigroup_sensr" in dataset_name:
        logger.info(f"model chossen amazon model")
        model_arch_params = config.simple_classification_dataset_model # don't need this expressive model. Simplify it!
    elif "adult"  in dataset_name or "dutch" in dataset_name or "celeb" in dataset_name:
        logger.info(f"model chossen adult model")
        print(f"model chossen adult model")
        model_arch_params = config.simple_classification_dataset_model  # don't need this expressive model. Simplify it!
        assert supervised_da == False
    elif "blog" in dataset_name or "encoded_bias_in_bios" in dataset_name:
        logger.info(f"model chossen blog model")
        print(f"model chossen blog model")
        model_arch_params = config.simple_classification_dataset_model_blog  # don't need this expressive model. Simplify it!
        assert supervised_da == False
    elif "encoded_emoji" in dataset_name:
        logger.info(f"model chossen encoded emoji model")
        print(f"model chossen blog model")
        model_arch_params = config.simple_classification_dataset_model_emoji  # don't need this expressive model. Simplify it!
        assert supervised_da == False

    # setting up seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    device = resolve_device() # if cuda: then cuda else cpu

    clean_text = clean_text_functional if use_clean_text else None
    max_length = max_length if max_length else None
    vocab = pickle.load(open(vocab_location, 'rb')) if vocab_location else None

    logger.info(f"initializing tokenizer: {tokenizer_type}")
    tokenizer = init_tokenizer(
        tokenizer=tokenizer_type,
        clean_text=clean_text,
        max_length=max_length
    )

    iterator_params = {
        'tokenizer': tokenizer,
        'artificial_populate': [],
        'pad_token': pad_token,
        'batch_size': batch_size,
        'is_regression': regression,
        'vocab': vocab,
        'use_adv_dataset': use_adv_dataset,
        'trim_data': trim_data,
        'sample_specific_class': sample_specific_class,
        'fairness_iterator': fairness_iterator,
        'fair_grad_newer': fair_grad, # Needs a larger validation split.
        'supervised_da': supervised_da
    }
    vocab, number_of_labels, number_of_aux_labels, iterators, other_data_metadata = \
        generate_data_iterators(dataset_name=dataset_name, **iterator_params)

    print(f"number of labels: {number_of_labels}")
    # need to pickle vocab. Same name as model save name but with additional "_vocab.pkl"
    # pickle.dump(vocab, open(model_save_name + '_vocab.pkl', 'wb'))

    # load pre-trained embeddings
    if use_pretrained_emb:
        print(f"reading pre-trained vector file from: {pre_trained_embeddings}")
        pretrained_embedding = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_embeddings)

        # infer input dim based on the pretrained_embeddings
        emb_dim = pretrained_embedding.vectors.shape[1]
    else:
        emb_dim = default_emb_dim


    output_dim = number_of_labels
    adv_output_dim = 2 # set this up automatically
    if len(vocab) > 10: # it is text and not just feature vector
        input_dim = len(vocab)
    else:
        for items in iterators[0]['train_iterator']:
            item = items
            break
        input_dim = item['input'].shape[1]

    model_name = copy.copy(model)

    print("here")

    if model == 'linear_adv':
        model_params = {
            'input_dim': input_dim,
            'output_dim': output_dim,

        }

        model_arch = model_arch_params
        model_arch['encoder']['input_dim'] = input_dim
        model_arch['main_task_classifier']['output_dim'] = output_dim
        model_arch['adv']['output_dim'] = adv_output_dim
        model_params = {
            'model_arch': model_arch,
            'noise_layer': noise_layer,
            'eps': eps,
            'device': device,
            'apply_noise_to_adv': apply_noise_to_adv
        }

        model = LinearAdv(model_params)

    elif model == 'linear_adv_encoded_emoji':
        model_params = {
            'input_dim': input_dim,
            'output_dim': output_dim,

        }

        model_arch = model_arch_params
        model_arch['encoder']['input_dim'] = input_dim
        model_arch['main_task_classifier']['output_dim'] = output_dim
        model_arch['adv']['output_dim'] = adv_output_dim
        model_params = {
            'model_arch': model_arch,
            'noise_layer': noise_layer,
            'eps': eps,
            'device': device,
            'apply_noise_to_adv': apply_noise_to_adv
        }

        model = LinearAdvEncodedEmoji(model_params)

    elif model == 'simple_linear':

        model_arch = {
            'encoder': {
                'input_dim': input_dim,
                'output_dim': output_dim
            }
        }

        model_params = {
            'model_arch': model_arch,
            'device': device
        }

        model = SimpleLinear(model_params)

    elif model == 'simple_non_linear':

        model_arch = {
            'encoder': {
                'input_dim': input_dim,
                'output_dim': output_dim
            }
        }

        model_params = {
            'model_arch': model_arch,
            'device': device
        }

        model = SimpleNonLinear(model_params)

    elif model == 'simple_non_linear_adv':
        model_arch = {
            'encoder': {
                'input_dim': input_dim,
                'output_dim': output_dim
            },
            'adv': {
                'number_of_layers': 2,
                'input_dim': 64,
                'hidden_dim':32,
                'output_dim':number_of_aux_labels,
                'dropout': 0.2
            }
        }

        model_params = {
            'model_arch': model_arch,
            'device': device
        }

        model = SimpleNonLinearAdv(model_params)


    # More stuff related to word embedding needs to be added here.
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    print(f"model is moved to {device}")

    opt_name = copy.copy(optimizer)
    # choosing the optimization function.
    if optimizer.lower() == 'adagrad':
        opt_fn = partial(torch.optim.Adagrad)
    elif optimizer.lower() == 'adam':
        opt_fn = partial(torch.optim.Adam)
    elif optimizer.lower() == 'sgd':
        opt_fn = partial(torch.optim.SGD)
    else:
        raise CustomError("no optimizer selected")
    optimizer = make_opt(model, opt_fn, lr=lr)



    # setup
    if use_lr_schedule:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                patience=4,
                factor=0.8,
                verbose=True
            )
    else:
        lr_scheduler = None

    # setting up loss function
    if number_of_labels == 1:
        criterion = nn.MSELoss()
        if use_wandb: wandb.log({'loss': 'MSEloss'})
    else:
        if fair_grad:
            criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            criterion = nn.CrossEntropyLoss()
        if use_wandb: wandb.log({'loss': 'CrossEntropy'})

    # setting up accuracy calculation function.
    accuracy_calculation_function = calculate_accuracy_regression if number_of_labels == 1 \
        else calculate_accuracy_classification

    # Fairness calculation function
    fairness_function = get_fairness_function(fairness_function)
    fairness_score_function = get_fairness_score_function(fairness_score_function)

    # Setup the training loop
    save_wrt_loss = False # if True; saves model wrt to lowest validation loss else highest training accuracy


    # Several of them are not useful.
    training_loop_params = {
        'is_adv': is_adv,
        'loss_aux_scale': adv_loss_scale,
        'is_regression': regression,
        'is_post_hoc': False,  # here the post-hoc has to be false
        'save_model': True,
        'seed': seed,
        'only_perturbate': only_perturbate,
        'mode_of_loss_scale': mode_of_loss_scale,
        'training_loop_type': training_loop_type,
        'hidden_l1_scale': hidden_l1_scale,
        'hidden_l2_scale': hidden_l2_scale,
        'return_hidden': hidden_loss,
        'reset_classifier': reset_classifier,
        'reset_adv': reset_adv,
        'encoder_learning_rate_second_phase': encoder_learning_rate_second_phase,
        'classifier_learning_rate_second_phase': classifier_learning_rate_second_phase,
        'eps': eps,
        'eps_scale': eps_scale,
        'fair_grad_newer': fair_grad,
        'reset_fairness': reset_fairness,
        'use_lr_schedule': use_lr_schedule,
        'lr_scheduler': lr_scheduler,
        'fairness_function': fairness_function,
        'fairness_score_function': fairness_score_function,
        'task': other_data_metadata['task'],
        'dataset_metadata': other_data_metadata,
        'save_wrt_loss': save_wrt_loss,
        'noise_layer': noise_layer,
        'calculate_leakage': calculate_leakage,
        'dataset': dataset_name,
        'clip_fairness': clip_fairness,
        'normalize_fairness': normalize_fairness,
        'opt_name': opt_name,
        'lr': lr,
        'apply_noise_to_adv': apply_noise_to_adv,
        'diverse_adv_lambda': diverse_adv_lambda
    }


    if fair_grad:
        # now we want to use baseline and adv setup here.
        #
        if hidden_l1_scale == 1: # A HACK. Not Clean solution!!!!
            training_loop_params['use_fair_grad'] = True
        else:
            training_loop_params['use_fair_grad'] = False
        best_test_acc, best_valid_acc, test_acc_at_best_valid_acc = fair_grad_training_loop(
            n_epochs=epochs,
            model=model,
            iterator=iterators[0],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            model_save_name=model_save_name,
            accuracy_calculation_function=accuracy_calculation_function,
            wandb=wandb,
            other_params=training_loop_params
        )
    elif supervised_da:
        best_test_acc, best_valid_acc, test_acc_at_best_valid_acc = k_fold_training_loop(
            n_epochs=epochs,
            model=model,
            iterator=iterators,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            model_save_name=model_save_name,
            accuracy_calculation_function=accuracy_calculation_function,
            wandb=wandb,
            other_params=training_loop_params
        )
    else:

        if diverse_adversary:

            best_test_acc, best_valid_acc, test_acc_at_best_valid_acc  = diverse_training_loop(
                n_epochs=epochs,
                model=model,
                iterator=iterators[0],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                model_save_name=model_save_name,
                accuracy_calculation_function=accuracy_calculation_function,
                wandb=wandb,
                other_params=training_loop_params
            )

        else:
            best_test_acc, best_valid_acc, test_acc_at_best_valid_acc  = simple_training_loop(
                n_epochs=epochs,
                model=model,
                iterator=iterators[0],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                model_save_name=model_save_name,
                accuracy_calculation_function=accuracy_calculation_function,
                wandb=wandb,
                other_params=training_loop_params
            )

        print(f"BEST Test Acc: {best_test_acc} ||"
              f" Actual Test Acc: {test_acc_at_best_valid_acc} || Best Valid Acc {best_valid_acc}")

        if use_wandb:
            wandb.config.update(
                {
                    'best_test_acc': best_test_acc,
                    'best_valid_acc': best_valid_acc,
                    'test_acc_at_best_valid_acc': test_acc_at_best_valid_acc
                }
            )
