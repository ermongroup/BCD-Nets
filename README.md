This folder contains the code for 'Scalable Variational Approaches for Bayesian Causal Discovery'.

# Installation
To install, use conda with
`conda env create -f environment.yml`.
If this fails for some reason, the key packages are
`jax jaxlib ott-jax cdt sklearn matplotlib optax dm-haiku tensorflow_probability torch wandb cython fuzzywuzzy python-Levenshtein sumu lingam`

You may have to recompile the cython module for the Hungarian algorithm by running 
`cython -3 mine.pyx` and then 
`g++ -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -o mine.so mine.c`
in the `c_modules` directory. 

# Running Experiments

Run with the `--use_wandb` flag to write results to a new weights and biases project. Otherwise, the results will be printed to stout.In `utils.py`  you may need to uncomment line 11 and replace your path to the `Rscript` binary

To run BCD Nets and GOLEM experiments in figure 1, for one random seed use arguments such as 
`python main.py -s 0 --n_data 100 --dim 32 --degree 1 --num_steps 30000 --do_ev_noise --sem_type linear-gauss --batch_size 256 --print_golem_solution --degree 1`

To run the baselines, run
`python main.py --eval_eid --run_baselines --n_data 100 --dim 32 --sem_type linear-gauss --only_baselines --degree 2  --do_ev_noise --n_baseline_seeds 3`

To run GOLEM, run
`python main.py --eval_eid --print_golem_solution --n_data 100 --dim 32 --sem_type linear-gauss --degree 2  --do_ev_noise --num_steps 10`

To run on the Sachs dataset, include the `--use_sachs` flag. 

