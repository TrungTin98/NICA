# NICA - Neural Imputation Cellular Automata

Authors: Trung-Tin Luu, Thanh-Binh Nguyen, Minh-Man Ngo

Pre-print: [Missing Data Imputation using Neural Cellular Automata](https://arxiv.org/abs/2509.00651)

This directory contains implementations of NICA framework for imputation on multiple datasets from [UCI](https://archive.ics.uci.edu/) and [Kaggle](https://www.kaggle.com/).

Example command to run the pipeline for training and imputing: `python main_imputation.py`.

You can also run with your own data and optional configuration. For example:

```bash
python main_imputation.py
--data_name iris --miss_rate 0.3
--n_versions 16 --batch_size 1024 --grow_steps 20
--alpha_1 10 --alpha_2 10 --dropout 0.5 --n_iters 1000
```
