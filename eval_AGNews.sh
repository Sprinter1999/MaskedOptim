
python main_fed_LNL.py \
--dataset AGNews \
--model fasttext \
--epochs 60 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 1.0 \
--test_bs 128 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_agnews_dirichlet10_sym04.txt

python main_fed_LNL.py \
--dataset AGNews \
--model fasttext \
--epochs 60 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 1.0 \
--test_bs 128 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_agnews_dirichlet10_pair04.txt

python main_fed_LNL.py \
--dataset AGNews \
--model fasttext \
--epochs 60 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.8 \
--partition dirichlet \
--dd_alpha 1.0 \
--test_bs 128 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_agnews_dirichlet10_sym08.txt

python main_fed_LNL.py \
--dataset AGNews \
--model fasttext \
--epochs 60 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition dirichlet \
--dd_alpha 1.0 \
--test_bs 128 \
--method maskedOptim | tee ./log_maskedOptim/maskedOptim_agnews_dirichlet10_mixed04.txt