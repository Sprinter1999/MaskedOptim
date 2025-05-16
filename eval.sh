mkdir log_maskedOptim
# source /etc/network_turbo #should be removed if you use your own computation device
pip install -r requirements.txt

python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.8 \
--partition dirichlet \
--dd_alpha 1.0 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_cifar10_sym08_dirichlet10.txt

python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 1.0 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_cifar10_pair04_dirichlet10.txt

python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 1.0 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_cifar10_sym04_dirichlet10.txt




python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 1.0 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_cifar10_mix04_dirichlet10.txt


python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.8 \
--partition dirichlet \
--dd_alpha 0.5 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_cifar10_sym08_dirichlet05.txt

python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 0.5 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_cifar10_pair04_dirichlet05.txt

python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 0.5 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_cifar10_sym04_dirichlet05.txt




python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 0.5 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_cifar10_mix04_dirichlet05.txt






