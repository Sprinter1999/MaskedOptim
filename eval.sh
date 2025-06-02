
python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 1.0 \
--test_bs 128 \
--method maskedOptim | tee maskedOptim_dirichlet10_sym04.txt
