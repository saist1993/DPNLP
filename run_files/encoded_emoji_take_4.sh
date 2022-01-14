cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv False --eps_list 100.0 --adv_start 0.1 --adv_end 2.0 --dataset_name encoded_emoji --log_name simple_baseline_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log1 &


# adversarial
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name adversarial_1_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv  &> log2 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name adversarial_2_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv  &> log3 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name adversarial_3_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv  &> log4 &


# diverse adversarial
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name diverse_adversarial_1_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.1 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log5 &


echo "first run running"



cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name diverse_adversarial_2_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.1 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log6 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name diverse_adversarial_3_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.1 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log7 &

cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name diverse_adversarial_4_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.2 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log8 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name diverse_adversarial_5_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.2 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log9 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name diverse_adversarial_6_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.2 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log10 &


echo "second run running"
wait
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name diverse_adversarial_7_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.3 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log11 &



cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name diverse_adversarial_8_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.3 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log12 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name diverse_adversarial_9_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.3 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log13 &

cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name diverse_adversarial_10_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.4 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log14 &


echo "third run running"


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name diverse_adversarial_11_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.4 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log15 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name diverse_adversarial_12_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.4 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log16 &

cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name diverse_adversarial_13_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.5 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log17 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name diverse_adversarial_14_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.5 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log18 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name diverse_adversarial_15_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.5 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log19 &


echo "fourth run running"
wait

cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name diverse_adversarial_16_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.6 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log20 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name diverse_adversarial_17_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.6 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log21 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name diverse_adversarial_18_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.6 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log22 &

cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name diverse_adversarial_19_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.7 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log23 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name diverse_adversarial_20_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.7 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log24 &


echo "fifth run running"


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name diverse_adversarial_21_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.7 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log25 &

cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name diverse_adversarial_22_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.8 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log26 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name diverse_adversarial_23_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.8 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log27 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name diverse_adversarial_24_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.8 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log28 &


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name diverse_adversarial_25_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.9 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log29 &


echo "sixth run running"
wait


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name diverse_adversarial_26_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.9 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log30 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name diverse_adversarial_27_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 0.9 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log31 &


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name diverse_adversarial_28_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 1.0 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log32 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name diverse_adversarial_29_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 1.0 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log33 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 100.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name diverse_adversarial_30_seed_10 --epochs 30 --noise_layer False --use_lr_schedule False --mode_of_loss_scale exp -diverse_adv_lambda 1.0 -diverse_adversary True -seed 10  --model linear_adv_encoded_emoji &> log34 &


echo "seventh run running"


# noise+ adversarial

cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 6.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name noise_adv_1_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log35 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 6.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name noise_adv_2_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log36 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 6.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name noise_adv_3_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log37 &


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 8.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name noise_adv_4_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log38 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 8.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name noise_adv_5_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log39 &


echo "eight run running"
wait


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 8.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name noise_adv_6_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log40 &


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 9.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name noise_adv_7_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log41 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 9.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name noise_adv_8_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log42 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 9.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name noise_adv_9_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log43 &

cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 10.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name noise_adv_10_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log44 &


echo "ninth run running"


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 10.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name noise_adv_11_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log45 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 10.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name noise_adv_12_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log46 &




cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 11.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name noise_adv_13_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log47 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 11.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name noise_adv_14_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log48 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 11.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name noise_adv_15_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log49 &


echo "tenth run running"
wait

cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 12.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name noise_adv_16_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log50 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 12.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name noise_adv_17_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log51 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 12.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name noise_adv_18_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log52 &


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 13.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name noise_adv_19_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log53 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 13.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name noise_adv_20_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log54 &


echo "eleven run running"



cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 13.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name noise_adv_21_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log55 &

cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 14.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name noise_adv_22_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log56 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 14.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name noise_adv_23_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log57 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 14.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name noise_adv_24_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log58 &


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 15.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name noise_adv_25_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log59 &

echo "twelve run running"

wait


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 15.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name noise_adv_26_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log60 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 15.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name noise_adv_27_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log61 &


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 16.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name noise_adv_28_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log62 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 16.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name noise_adv_29_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log63 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 16.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name noise_adv_30_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log64 &


echo "thirteen run running"



cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 20.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name noise_adv_31_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log65 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 20.0 --adv_start 1.0 --adv_end 2.0 --dataset_name encoded_emoji --log_name noise_adv_32_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log66 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv True --eps_list 20.0 --adv_start 2.0 --adv_end 3.0 --dataset_name encoded_emoji --log_name noise_adv_33_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log67 &




cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv False --eps_list 6.0 8.0 9.0 10.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name only_noise_1_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log68 &
cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv False --eps_list 11.0 12.0 13.0 14.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name only_noise_2_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log69 &


cd ~/codes/DPNLP/src/; python hyperparam_runner.py --is_adv False --eps_list 15.0 16.0 20.0 --adv_start 0.1 --adv_end 1.0 --dataset_name encoded_emoji --log_name only_noise_3_seed_10 --epochs 45 --noise_layer True --use_lr_schedule False --mode_of_loss_scale exp -seed 10  --model linear_adv &> log70 &

echo "fourteenth running! Last"

wait

echo "done with 10 seed."

# unconstrained

