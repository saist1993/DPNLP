# File where everything begins.

# Library Imports
import copy
import click
import random
import pickle
import gensim
import logging
import numpy as np
from typing import Optional, Callable

# Custom Imports
import config
from utils.misc import *
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
         sample_specific_class:bool
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
        model_params = config.amazon_model

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
        'sample_specific_class': sample_specific_class
    }
    vocab, number_of_labels, number_of_aux_labels, iterators, other_data_metadata = \
        generate_data_iterators(dataset_name=dataset_name, **iterator_params)

    print(f"number of labels: {number_of_labels}")
    # need to pickle vocab. Same name as model save name but with additional "_vocab.pkl"
    pickle.dump(vocab, open(model_save_name + '_vocab.pkl', 'wb'))

    # load pre-trained embeddings
    if use_pretrained_emb:
        print(f"reading pre-trained vector file from: {pre_trained_embeddings}")
        pretrained_embedding = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_embeddings)

        # infer input dim based on the pretrained_embeddings
        emb_dim = pretrained_embedding.vectors.shape[1]
    else:
        emb_dim = default_emb_dim


    output_dim = number_of_labels
    if len(vocab) > 10: # it is text and not just feature vector
        input_dim = len(vocab)
    else:
        for items in iterators[0]['train_iterators']:
            item = items
            break
        input_dim = item[1].shape[1]

    model_name = copy.copy(model)

    if model == 'linear_adv':
        model_params = {
            'input_dim': input_dim,
            'output_dim': output_dim,
        }





