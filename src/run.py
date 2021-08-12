import main
import click
from typing import Optional

@click.command()
@click.option('-embedding', '--emb_dim', type=int, default=300)
@click.option('-spacy', '--spacy_model', type=str, default="en_core_web_sm", help="the spacy model used for tokenization. This might not be suitable for twitter and other noisy use cases ")
@click.option('-seed', '--seed', type=int, default=1234)
@click.option('-data', '--dataset_name', type=str, default='wiki_debias', help='the first half (wiki) is the name of the dataset, and the second half (debias) is the specific kind to use')
@click.option('-bs', '--batch_size', type=int, default=512)
@click.option('-pad', '--pad_token', type=str, default='<pad>')
@click.option('-unk', '--unk_token', type=str, default='<unk>')
@click.option('-embeddings', '--pre_trained_embeddings', type=str, default='../../bias-in-nlp/different_embeddings/simple_glove_vectors.vec') # work on this.
@click.option('-save_model_as', '--model_save_name', type=str, default='bilstm.pt')
@click.option('-model', '--model', type=str, default='bilstm')
@click.option('-is_regression', '--regression', type=bool, default=True, help='if regression then sentiment/toxicity is a continous value else classification.')
@click.option('-tokenizer', '--tokenizer_type', type=str, default="spacy", help='currently available: tweet, spacy, simple')
@click.option('-clean_text', '--use_clean_text', type=bool, default=False)
@click.option('-max_len', '--max_length', type=int, default=None)
@click.option('-epochs', '--epochs', type=int, default=30)
@click.option('-learnable_embeddings', '--learnable_embeddings', type=bool, default=False)
@click.option('-vocab_location', '--vocab_location', type=bool, default=False, help="file path location. Generally used while testing to load a vocab. Type is incorrect.")
@click.option('-is_adv', '--is_adv', type=bool, default=False, help="if True; adds an adversarial loss to the mix.")
@click.option('-adv_loss_scale', '--adv_loss_scale', type=float, default=0.5, help="sets the adverserial scale (lambda)")
@click.option('-use_pretrained_emb', '--use_pretrained_emb', type=bool, default=True, help="uses pretrianed if true else random")
@click.option('-default_emb_dim', '--default_emb_dim', type=int, default=100, help="uses pretrianed if true else random")
@click.option('-save_test_pred', '--save_test_pred', type=bool, default=False, help="has very specific use case: only works with adv_bias_in_bios")
@click.option('-noise_layer', '--noise_layer', type=bool, default=False, help="used for diff privacy. For now, not implemented")
@click.option('-eps', '--eps', type=float, default=1.0, help="privacy budget")
@click.option('-is_post_hoc', '--is_post_hoc', type=bool, default=False, help="trains a post-hoc classifier")
@click.option('-train_main_model', '--train_main_model', type=bool, default=True, help="If false; only trains post-hoc classifier")
@click.option('-use_wandb', '--use_wandb', type=bool, default=False, help="make sure the project is configured to use wandb")
@click.option('-config_dict', '--config_dict', type=str, default="simple", help="which config to use")
@click.option('-experiment_name', '--experiment_name', type=str, default="NA", help="name of group of experiment")
@click.option('-only_perturbate', '--only_perturbate', type=bool, default=False, help="If True; only trains on perturbate phase. Like a vanilla DAAN")
@click.option('-mode_of_loss_scale', '--mode_of_loss_scale', type=str, default="constant", help="constant/linear. The way adv loss scale to be increased with epochs during gradient reversal mode.")
@click.option('-training_loop_type', '--training_loop_type', type=str, default="three_phase_custom", help="three_phase/three_phase_custom are the two options. Only works with is_adv true")
@click.option('-hidden_loss', '--hidden_loss', type=bool, default=False, help="if true model return hidden. Generally used in case of adding a L1/L2 regularization over hidden")
@click.option('-hidden_l1_scale', '--hidden_l1_scale', type=float, default=0.5, help="scaling l1 loss over hidden")
@click.option('-hidden_l2_scale', '--hidden_l2_scale', type=float, default=0.5, help="scaling l2 loss over hidden")
@click.option('-reset_classifier', '--reset_classifier', type=bool, default=False, help="resets classifier in the third Phase of adv training.")
@click.option('-reset_adv', '--reset_adv', type=bool, default=False, help="resets adv in the third Phase of adv training.")
@click.option('-encoder_learning_rate_second_phase', '--encoder_learning_rate_second_phase', type=float, default=0.01, help="changes the learning rate of encoder (embedder) in second phase")
@click.option('-classifier_learning_rate_second_phase', '--classifier_learning_rate_second_phase', type=float, default=0.01, help="changes the learning rate of main task classifier in second phase")
@click.option('-trim_data', '--trim_data', type=bool, default=False, help="decreases the trainging data in  bias_in_bios to 15000")
@click.option('-eps_scale', '--eps_scale', type=str, default="constant", help="constant/linear. The way eps should decrease with iteration.")
@click.option('-optimizer', '--optimizer', type=str, default="adam", help="only works when adv is True")
@click.option('-lr', '--lr', type=float, default=0.01, help="main optimizer lr")
@click.option('-fair_grad', '--fair_grad', type=bool, default=False, help="implements the fair sgd and training loop")
@click.option('-reset_fairness', '--reset_fairness', type=bool, default=False, help="resets fairness every epoch. By default fairness is just added")
@click.option('-use_adv_dataset', '--use_adv_dataset', type=bool, default=True, help="if True: output includes aux")
@click.option('-use_lr_schedule', '--use_lr_schedule', type=bool, default=True, help="if True: lr schedule is implemented. Note that this is only for simple trainign loop and not for three phase ones.")
@click.option('-fairness_function', '--fairness_function', type=str, default='equal_odds', help="the fairness measure to implement while employing fairgrad.")
@click.option('-fairness_score_function', '--fairness_score_function', type=str, default='grms', help="The fairness score function.")
@click.option('-sample_specific_class', '--sample_specific_class', type=bool, default=False, help="samples only specific classes. Specified in create_data.BiasinBiosSimpleAdv class")
@click.option('-calculate_leakage', '--calculate_leakage', type=bool, default=False, help="leakage from the test set.")
@click.option('-clip_fairness', '--clip_fairness', type=bool, default=True, help="Clip fairness to max of 1.0")
@click.option('-normalize_fairness', '--normalize_fairness', type=bool, default=False, help="normalizes fairness before multiplying with gradients.")
@click.option('-fairness_iterator', '--fairness_iterator', type=str, default="train", help="train/custom_1/custom_2 etc.")
@click.option('-supervised_da', '--supervised_da', type=bool, default=False, help="Does supervised domain adapatation if true.")
def run(emb_dim:int,
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
         supervised_da:bool
         ):
    main.main(emb_dim,
             spacy_model,
             seed,
             dataset_name,
             batch_size,
             pad_token,
             unk_token,
             pre_trained_embeddings,
             model_save_name,
             model,
             regression,
             tokenizer_type,
             use_clean_text,
             max_length,
             epochs,
             learnable_embeddings,
             vocab_location,
             is_adv,
             adv_loss_scale,
             use_pretrained_emb,
             default_emb_dim,
             save_test_pred,
             noise_layer,
             eps,
             is_post_hoc,
             train_main_model,
             use_wandb,
             config_dict,
             experiment_name,
             only_perturbate,
             mode_of_loss_scale,
             training_loop_type,
             hidden_loss,
             hidden_l1_scale,
             hidden_l2_scale,
             reset_classifier,
             reset_adv,
             encoder_learning_rate_second_phase,
             classifier_learning_rate_second_phase,
             trim_data,
             eps_scale,
             optimizer,
             lr,
             fair_grad,
             reset_fairness,
             use_adv_dataset,
             use_lr_schedule,
             fairness_function,
             fairness_score_function,
             sample_specific_class,
             calculate_leakage,
             clip_fairness,
             normalize_fairness,
             fairness_iterator,
             supervised_da
             )

if __name__ == '__main__':
    run()