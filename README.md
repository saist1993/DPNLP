# Fair NLP Models with Differentially Private Text Encoders - FEDERATE

This is a codebase corresponding to FEDERATE, an approach that combines ideas from differential privacy and adversarial learning to learn private text representations which also induces fairer models. It is still work in progress, and things might break as it has not been tested in various development environment. 

### Folder Structure

- **datasets**: All datasets are stored here. Please download the datasets from this [URL](https://drive.google.com/uc?id=1ZmUE-g6FmzPPbZyw3EOki7z4bpzbKGWk). 
- **logs**: All raw logs for reproducing expeiment results are present in this folder. Check ```parse_log``` folder in ```src``` for parsing and interpreting these logs.
- **run_files**: Shell files to run over various datasets including hyper-param search. 
- **src**: All the source file (python).


### Requirements 

All experiments employed ```python 3.8```. Please install other python dependency via ```requirements.txt``` file. 

### Example run 

Following code runs Federate over Bias in Bios dataset with a specific hyper parameters sweep. 

```
cd src/; python hyperparam_runner.py --is_adv True --eps_list 6.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_bias_in_bios --log_name noise_adv_1_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv
```

Various aspects of the codebase can be controlled via arguments. To find the complete list please refer to file ```src/run.py```
