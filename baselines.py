import sumu  # type: ignore
from dag_utils import (
    SyntheticDataset,
    process_sachs,
    get_sachs_ground_truth,
)
from cdt.causality.graph import GIES, GES, LiNGAM, PC
from utils import (
    eval_W_non_ev,
    eval_W_ev,
    auroc,
    from_W,
    get_variance,
    get_variances,
    random_str,
    fit_known_edges,
)
from pandas import DataFrame
import networkx as nx
from cdt.metrics import SHD_CPDAG
from lingam import DirectLiNGAM
from dag_utils import dagify
import jax.numpy as jnp
import numpy as np
from metrics import intervention_distance, ensemble_intervention_distance
import time
import jax.random as rnd
import os
import sys
from tqdm import tqdm
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from typing import cast, Optional
from sklearn.utils import resample

subset_size = 50
graphical_lasso_iters = 10_000  # default is 100


def run_gadget(
    x_train, x_test, ground_truth_W, ev=True, subsample=subset_size, n_sumu_iters=50_000
):
    _, d = x_train.shape
    datapath = f"./tmp/for_summu_{random_str()}.csv"
    np.savetxt(datapath, x_train, delimiter=" ")
    time.sleep(5)  # Give time to save data properly
    if d < 16:
        K = d - 1
    else:
        K = 15
    data = sumu.Data(datapath, discrete=False)
    t0 = time.time()
    params = {
        "data": data,
        "scoref": "bge",  # Or "bdeu" for discrete data.
        "ess": 10,  # If using BDeu.
        "max_id": -1,  # Max indegree, -1 for none.
        "K": K,  # Number of candidate parents per variable (< n).
        "d": 3,  # Max size for parent sets not constrained to candidates.
        "cp_algo": "greedy-lite",  # Algorithm for finding the candidate parents.
        "mc3_chains": 48,  # Number of parallel Metropolis coupled Markov chains.
        "burn_in": int(
            0.5 * n_sumu_iters
        ),  # Number of burn-in iterations in the chain.
        "iterations": n_sumu_iters,  # Number of iterations after burn-in.
        "thinning": 10,
    }  # Sample a DAG at every nth iteration.
    g = sumu.Gadget(**params)

    # dags is a list of tuples, where the first element is an int encoding a node
    # and the second element is a (possibly empty) tuple of its parents.
    dags, _ = g.sample()
    print(f"Took {time.time() - t0}s to do {n_sumu_iters} steps")

    # Causal effect computations only for continuous data.
    # dags are first converted to adjacency matrices.
    dags = [sumu.bnet.family_sequence_to_adj_mat(dag).T for dag in dags]

    # All pairwise causal effects for each sampled DAG.
    # causal_effects[i] : effects for ith DAG,
    # where the the first n-1 values represent the effects from variable 1 to 2, ..., n,
    # the following n-1 values represent the effects from variable 2 to 1, 3, ..., n, etc.
    # causal_effects = sumu.beeps(dags, data)
    Ws = np.array(dags)

    # np.save(f"./{d}_{seed}_{sem_type}_dags.npy", numpy_dags)

    # ground_truth_W = np.load(f"{dim}_{seed}_{sem_type}_gt.npy")
    # adjs = np.load(f"./{dim}_{ssed}_dags.npy")
    random_idxs = np.random.choice(len(Ws), size=(subsample))
    Ws = Ws[random_idxs]
    out_Ws = []
    for W in Ws:
        W = fit_known_edges(
            W, x_train, max_iters=20_000, lr=1e-3, verbose=True, lambda_1=1e-5
        )
        est_W_clipped = np.where(np.abs(W) > 0.3, W, 0)
        out_Ws.append(est_W_clipped[None, ...])
    Ws = jnp.concatenate(out_Ws, axis=0)
    out_stats = eval_W_samples(
        Ws,
        x_train,
        ground_truth_W,
        jnp.ones(d),
        jnp.ones(d),
        ev,
        do_shd_c=True,
        do_sid=True,
        subsample=subset_size,
        x_prec=None,
        filename="tmp/gadget_roc.png",
    )
    est_noises_var = []
    print("Fitting gadget edge coefficients:")
    for W in Ws:
        if ev:
            est_noises_var.append(get_variance(from_W(W, d), x_train)[None, ...])
        else:
            est_noises_var.append(get_variances(from_W(W, d), x_train)[None, ...])
    est_noises_var = jnp.concatenate(est_noises_var)
    return (
        Ws,
        out_stats,
        out_stats["shd_c"],
        out_stats["shd"],
        out_stats["tpr"],
        out_stats["fpr"],
        out_stats["fdr"],
        jnp.sqrt(est_noises_var),
    )


def ghoshal(X, ground_truth_W):
    # No code available to our knowledge, have to re-implement this method
    _, d = X.shape
    try:
        clf = GraphicalLassoCV(max_iter=graphical_lasso_iters)
        clf.fit(X)
        O = clf.get_precision()
        O = cast(jnp.ndarray, O)
        O_empirical = np.linalg.pinv(jnp.cov(X.T))
        ground_truth_O = (jnp.eye(d) - ground_truth_W) @ (jnp.eye(d) - ground_truth_W).T
        O_dist = jnp.sqrt(jnp.mean((O - ground_truth_O) ** 2))
        empirical_dist = jnp.sqrt(jnp.mean((O_empirical - ground_truth_O) ** 2))
        if empirical_dist < O_dist:
            O = O_empirical
    except:
        graphical_lasso_success = False
        O_empirical = np.linalg.pinv(jnp.cov(X.T))
        O = O_empirical

    # A minor hack here: sometimes GraphicalLassoCV completely fails
    # to find a good precision, since we're pretty underdetermined. In
    # this case we're better off just using the unregularized
    # empirical estimate. In the interest of giving a stronger
    # baseline, we'll use the real W to choose when to do this or not,
    # since it might be that a better precision estimator would do
    # better here

    B = np.zeros((d, d))
    D = np.eye(d)
    for d in range(d):
        i = np.argmin(np.diag(O * D))
        B[i, :] = -O[i, :] / O[i, i]
        B[i, i] = 0
        O = O - (O[:, i][:, None] @ O[i, :][None, :]).T / O[i, i]
        O[i, i] = np.inf
    return O, B


def run_ghoshal(Xs, x_test, ground_truth_W, do_ev_noise):
    t0 = time.time()
    _, W = ghoshal(Xs, ground_truth_W)
    print(f"Found Ghoshal estimator in {time.time() - t0}")
    dim = len(W)
    W = dagify(jnp.where(jnp.abs(W) < 0.3, 0.0, W)).T
    W = np.where(np.abs(W) > 0.3, W, 0)

    if do_ev_noise:
        ghoshal_eval = eval_W_ev(W, ground_truth_W, np.ones(dim), 0.3, x_test, None)
        est_noise_var = get_variance(from_W(W, dim), Xs)
    else:
        ghoshal_eval = eval_W_non_ev(W, ground_truth_W, np.ones(dim), 0.3, x_test, None)
        est_noise_var = get_variances(from_W(W, dim), Xs)
    return (
        W,
        ghoshal_eval["shd_c"],
        ghoshal_eval["shd"],
        ghoshal_eval["tpr"],
        ghoshal_eval["fpr"],
        ghoshal_eval["fdr"],
        jnp.sqrt(est_noise_var),
    )


def run_peters(Xs, x_test, ground_truth_W, do_ev_noise):
    if do_ev_noise:
        eval_W_fn = eval_W_ev
    else:
        eval_W_fn = eval_W_non_ev

    _, d = Xs.shape
    print(os.getcwd())
    datapath = f"tmp/for_peters_{random_str()}.csv"
    outpath = f"tmp/for_peters_{random_str()}_out.csv"
    np.savetxt(
        datapath,
        Xs,
        delimiter=",",
        header=", ".join([f"X_{i}" for i in range(d)]),  # ignore
    )
    time.sleep(5)  # ensure the data is saved properly
    exit_code = os.system(
        f"cd codeforGDSEEV; Rscript load_and_infer.r ../{datapath} ../{outpath}"
    )
    if exit_code != 0:
        print(f"Some error occured in calling R, exit code {exit_code}")
        return jnp.ones((d, d)) * jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan
    W = np.genfromtxt(outpath, delimiter=",").T
    # Uses different convention so we need to take transpose
    dim = len(W)
    W = dagify(jnp.where(jnp.abs(W) < 0.3, 0.0, W)).T
    # Peters gives us the binary matrix, now we fit the actual coefficients
    # This is easily done via gradient descent
    print("fitting Peters edge coefficients")
    W_coeff = fit_known_edges(
        W, Xs, max_iters=20_000, lr=1e-3, verbose=True, lambda_1=1e-5
    )
    est_W_clipped = np.where(np.abs(W_coeff) > 0.3, W_coeff, 0)
    peters_eval = eval_W_fn(W, ground_truth_W, np.ones(dim), 0.3, x_test, None)

    if do_ev_noise:
        est_noise_var = get_variance(from_W(est_W_clipped, d), Xs)
    else:
        est_noise_var = get_variances(from_W(est_W_clipped, d), Xs)
    return (
        est_W_clipped,
        peters_eval["shd_c"],
        peters_eval["shd"],
        peters_eval["tpr"],
        peters_eval["fpr"],
        peters_eval["fdr"],
        jnp.sqrt(est_noise_var),
    )


def to_array_rstyle(x, dim):
    """Given an array, write to adjacency matrix, 
    indexed in the style from bc_mcmc, i.e. opposite to 
    triu_indices. Slow but we only use it for baselines"""
    W = np.zeros((dim, dim))
    row = 0
    col = 1
    for k in x:
        if row >= col:
            col += 1
            row = 0
        W[row, col] = k
        row += 1
    return W


def eval_W_samples(
    Ws,
    Xs,
    ground_truth_W,
    predicted_noise,
    ground_truth_noise,
    do_ev_noise,
    do_shd_c,
    do_sid,
    subsample=subset_size,
    x_prec=None,
    filename: Optional[str] = None,
):
    random_idxs = np.random.choice(len(Ws), size=(subsample))
    Ws = Ws[random_idxs]
    if do_ev_noise:
        eval_W_fn = eval_W_ev
    else:
        eval_W_fn = eval_W_non_ev

    def sample_stats(W):
        stats = eval_W_fn(
            W,
            ground_truth_W,
            ground_truth_noise,
            0.3,
            Xs,
            predicted_noise,
            provided_x_prec=x_prec,
            do_shd_c=do_shd_c,
            do_sid=do_sid,
        )
        return stats

    stats = sample_stats(Ws[0])
    stats = {key: [stats[key]] for key in stats}
    for _, W in tqdm(enumerate(Ws[1:]), total=len(Ws) - 1):
        new_stats = sample_stats(W)
        for key in new_stats:
            stats[key] = stats[key] + [new_stats[key]]

    out_stats = {key: np.mean(stats[key]) for key in stats}
    out_stats["auroc"] = auroc(Ws, ground_truth_W, 0.3)
    return out_stats


def run_bc_mcmc(Xs, ground_truth_W, do_ev_noise):
    x_prec = np.linalg.inv(jnp.cov(Xs.T))

    _, d = Xs.shape
    datapath = f"./tmp/for_bc_mcmc_{random_str()}.csv"
    outpath = f"./tmp/for_bc_mcmc_{random_str()}_out.csv"
    np.savetxt(
        datapath,
        Xs,
        delimiter=",",
        header=", ".join([f"X_{i}" for i in range(d)]),  # ignore
    )
    time.sleep(10)  # Give time to make sure data saved properly
    exit_code = os.system(f"Rscript bc_mcmc.r {datapath} {outpath}")
    if exit_code != 0:
        print(f"Some error occured in calling R, exit code {exit_code}")
        Ws = jnp.ones((20, d, d)) * jnp.nan
    else:
        samples = []
        for line in open(outpath, "r"):
            processed_line = line.strip().strip('"')
            np_line = np.fromstring(" ".join(list(processed_line)), sep=" ")
            samples.append(to_array_rstyle(np_line, d))

        Ws = jnp.vstack([sample[None, ...] for sample in samples])

    random_idxs = np.random.choice(len(Ws), size=(subset_size))
    Ws = Ws[random_idxs]

    out_stats = eval_W_samples(
        Ws,
        Xs,
        ground_truth_W,
        jnp.ones(d),
        jnp.ones(d),
        do_ev_noise,
        do_shd_c=True,
        do_sid=True,
        subsample=subset_size,
        x_prec=x_prec,
    )
    return (
        out_stats,
        out_stats["shd_c"],
        out_stats["shd"],
        out_stats["tpr"],
        out_stats["fpr"],
        out_stats["fdr"],
    )


def run_GES(x_train, ground_truth_W):
    obj = GES()
    output = obj.create_graph_from_data(data=DataFrame(x_train))
    shd_c = SHD_CPDAG(output, nx.DiGraph(np.array(np.abs(ground_truth_W) > 0)))
    W = nx.to_numpy_array(output)
    W = dagify(jnp.where(jnp.abs(W) < 0.3, 0.0, W)).T
    return W, shd_c


def run_LiNGAM(x_train, x_test, ground_truth_W, ev=True):
    _, d = x_train.shape
    obj = LiNGAM()
    try:
        output = obj.create_graph_from_data(data=DataFrame(x_train))
    except:
        print("lingam failed")
        # Not sure why this fails but it seems to be particularly bad
        # for high-dimension, low-data cases
        W = jnp.ones((d, d)) * jnp.nan
        return (
            W,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    else:
        shd_c = SHD_CPDAG(output, nx.DiGraph(np.array(np.abs(ground_truth_W).T > 0)))
        W = nx.to_numpy_array(output)
        W = dagify(jnp.where(jnp.abs(W) < 0.3, 0.0, W)).T
    est_W_clipped = np.where(np.abs(W) > 0.3, W, 0)
    if ev:
        lingam_eval = eval_W_ev(
            est_W_clipped, ground_truth_W, np.ones(d), 0.3, x_test, None
        )
        est_noise_var = get_variance(from_W(est_W_clipped, d), x_train)
    else:
        lingam_eval = eval_W_non_ev(
            est_W_clipped, ground_truth_W, np.ones(d), 0.3, x_test, None
        )
        est_noise_var = get_variances(from_W(est_W_clipped, d), x_train)

    return (
        W,
        lingam_eval["shd_c"],
        lingam_eval["shd"],
        lingam_eval["tpr"],
        lingam_eval["fpr"],
        lingam_eval["fdr"],
        jnp.sqrt(est_noise_var),
    )


def run_bootstrapped_LiNGAM(
    x_train,
    x_test,
    ground_truth_W,
    ev=True,
    bootstrap_frac=0.95,
    bootstrap_iters=20,
    rng_key=rnd.PRNGKey(0),
):

    data_size = int(bootstrap_frac * len(x_train))
    random_states = rnd.split(rng_key, bootstrap_iters)[:, 0]

    stats_out = []
    for k in range(bootstrap_iters):
        if k % 5 == 0:
            print(f"At bootstrap iter {k}")
        data = resample(
            x_train,
            replace=True,
            n_samples=data_size,
            random_state=int(random_states[k]),
        )
        LiNGAM_outs = run_LiNGAM(data, x_test, ground_truth_W, ev)
        LiNGAM_out_dict = {
            key: LiNGAM_outs[i]
            for i, key in enumerate(["W", "shd_c", "shd", "tpr", "fpr", "fdr"])
        }
        stats_out.append(LiNGAM_out_dict)
    # Get a set of possible Ws here
    out_dict = {key: [stat_out[key] for stat_out in stats_out] for key in stats_out[0]}
    all_Ws = out_dict["W"]
    all_Ws = jnp.vstack([W[None, ...] for W in all_Ws])
    mean_dict = {key: np.mean(out_dict[key]) for key in out_dict}
    mean_dict["auroc"] = auroc(all_Ws, ground_truth_W, 0.3)
    return all_Ws, mean_dict


def run_PC(x_train, x_test, ground_truth_W, ev=True):
    n, dim = x_train.shape
    obj = PC()
    output = obj.create_graph_from_data(data=DataFrame(x_train))
    shd_c = SHD_CPDAG(output, nx.DiGraph(np.array(np.abs(ground_truth_W) > 0)))
    W = nx.to_numpy_array(output)
    W = dagify(jnp.where(jnp.abs(W) < 0.3, 0.0, W)).T
    if ev:
        pc_eval = eval_W_ev(
            W, ground_truth_W, np.ones(dim), 0.3, x_test, None, get_wasserstein=False
        )
    else:
        pc_eval = eval_W_non_ev(
            W, ground_truth_W, np.ones(dim), 0.3, x_test, None, get_wasserstein=False
        )
    # print("pc", pc_eval)
    # print("direct shd_c:", shd_c)
    return (
        W,
        shd_c,
    )


def run_DirectLiNGAM(x_train, x_test, ground_truth_W, ev=True):
    _, dim = x_train.shape
    model = DirectLiNGAM()
    model.fit(x_train)
    W = model.adjacency_matrix_
    W = dagify(jnp.where(jnp.abs(W) < 0.3, 0.0, W)).T
    est_W_clipped = np.where(np.abs(W) > 0.3, W, 0)
    if ev:
        direct_lingam_eval = eval_W_ev(
            W, ground_truth_W, np.ones(dim), 0.3, x_test, None, get_wasserstein=False
        )
        est_noise_var = get_variance(from_W(est_W_clipped, dim), x_train)

    else:
        direct_lingam_eval = eval_W_non_ev(
            W, ground_truth_W, np.ones(dim), 0.3, x_test, None, get_wasserstein=False
        )
        est_noise_var = get_variances(from_W(est_W_clipped, dim), x_train)

    # print("direct_lingam: ", direct_lingam_eval)
    return (
        W,
        direct_lingam_eval["shd_c"],
        direct_lingam_eval["shd"],
        direct_lingam_eval["tpr"],
        direct_lingam_eval["fpr"],
        direct_lingam_eval["fdr"],
        jnp.sqrt(est_noise_var),
    )


def run_all_baselines(
    seeds,
    n_data,
    dim,
    sem_type,
    degree,
    do_ev_noise,
    sachs=False,
    fast_baselines=False,
    n_sumu_iters=50_000,
):
    n_seeds = len(seeds)
    results_df = DataFrame()
    t0 = time.time()
    for i, random_seed in enumerate(seeds):
        print(f"Starting seed {i}")
        np.random.seed(random_seed)
        if do_ev_noise:
            log_sigma_W = jnp.zeros(dim)
        else:
            log_sigma_W = np.random.uniform(low=0, high=jnp.log(2), size=(dim,))
        if sachs:
            dim = 11
            rng_key_1, rng_key_2 = rnd.split(rnd.PRNGKey(random_seed))
            Xs = process_sachs(
                center=True, normalize=True, n_data=n_data, rng_key=rng_key_1
            )
            test_Xs = process_sachs(center=True, normalize=True, rng_key=rng_key_2)
            ground_truth_W = get_sachs_ground_truth()
            n_data = len(Xs)
            ground_truth_sigmas = jnp.ones(dim) * jnp.nan
        else:
            sd = SyntheticDataset(
                n=n_data,
                d=dim,
                graph_type="erdos-renyi",
                degree=2 * degree,
                sem_type=sem_type,
                dataset_type="linear",
            )
            ground_truth_W = sd.W
            ground_truth_P = sd.P
            Xs = sd.simulate_sem(
                ground_truth_W,
                n_data,
                sd.sem_type,
                w_range=None,
                noise_scale=None,
                dataset_type="linear",
                W_2=None,
                sigmas=jnp.exp(log_sigma_W),
            )

            ground_truth_sigmas = jnp.ones(dim) * jnp.exp(log_sigma_W)
            test_Xs = sd.simulate_sem(
                ground_truth_W,
                sd.n,
                sd.sem_type,
                sd.w_range,
                None,
                sd.dataset_type,
                sd.W_2,
                sigmas=jnp.exp(log_sigma_W),
            )

        print("Running GES")
        GES_W, ges_shd_c = run_GES(Xs, ground_truth_W)
        print("Running LiNGAM")
        (
            LiNGAM_W,
            lingam_shd_c,
            lingam_shd,
            lingam_tpr,
            lingam_fpr,
            lingam_fdr,
            lingam_noise,
        ) = run_LiNGAM(Xs, test_Xs, ground_truth_W, do_ev_noise)

        print("Running PC")
        PC_W, pc_shd_c = run_PC(Xs, test_Xs, ground_truth_W, do_ev_noise)

        print("Running DirectLiNGAM")
        (
            DirectLiNGAM_W,
            dlingam_shd_c,
            dlingam_shd,
            dlingam_tpr,
            dlingam_fpr,
            dlingam_fdr,
            dlingam_noise,
        ) = run_DirectLiNGAM(Xs, test_Xs, ground_truth_W, do_ev_noise)

        print("Running Ghoshal")
        (
            ghoshal_W,
            ghoshal_shd_c,
            ghoshal_shd,
            ghoshal_tpr,
            ghoshal_fpr,
            ghoshal_fdr,
            ghoshal_noise,
        ) = run_ghoshal(Xs, test_Xs, ground_truth_W, do_ev_noise)

        print("Running Peters")
        (  # type: ignore
            peters_W,
            peters_shd_c,
            peters_shd,
            peters_tpr,
            peters_fpr,
            peters_fdr,
            peters_noise,
        ) = run_peters(Xs, test_Xs, ground_truth_W, do_ev_noise)

        try:
            lingam_eid = intervention_distance(
                ground_truth_W, LiNGAM_W, ground_truth_sigmas, lingam_noise, sem_type
            )
        except:
            lingam_eid = np.nan
        try:
            dlingam_eid = intervention_distance(
                ground_truth_W,
                DirectLiNGAM_W,
                ground_truth_sigmas,
                dlingam_noise,
                sem_type,
            )
        except:
            dlingam_eid = np.nan

        try:
            peters_eid = intervention_distance(
                ground_truth_W, peters_W, ground_truth_sigmas, peters_noise, sem_type
            )
        except:
            peters_eid = np.nan

        try:
            ghoshal_eid = intervention_distance(
                ground_truth_W, ghoshal_W, ground_truth_sigmas, ghoshal_noise, sem_type
            )
        except:
            ghoshal_eid = np.nan

        # Write fast baselines
        results_df.loc[i, "ges_shd_c"] = ges_shd_c
        results_df.loc[i, "lingam_shd_c"] = lingam_shd_c
        results_df.loc[i, "dlingam_shd_c"] = dlingam_shd_c
        results_df.loc[i, "pc_shd_c"] = pc_shd_c
        results_df.loc[i, "ghoshal_shd_c"] = ghoshal_shd_c

        results_df.loc[i, "lingam_shd"] = lingam_shd
        results_df.loc[i, "dlingam_shd"] = dlingam_shd
        results_df.loc[i, "ghoshal_shd"] = ghoshal_shd
        results_df.loc[i, "peters_shd"] = peters_shd

        results_df.loc[i, "lingam_tpr"] = lingam_tpr
        results_df.loc[i, "dlingam_tpr"] = dlingam_tpr
        results_df.loc[i, "ghoshal_fpr"] = ghoshal_fpr
        results_df.loc[i, "peters_tpr"] = peters_tpr

        results_df.loc[i, "lingam_fpr"] = lingam_fpr
        results_df.loc[i, "dlingam_fpr"] = dlingam_fpr
        results_df.loc[i, "ghoshal_fpr"] = ghoshal_fpr
        results_df.loc[i, "peters_fpr"] = peters_fpr

        results_df.loc[i, "lingam_fdr"] = lingam_fdr
        results_df.loc[i, "dlingam_fdr"] = dlingam_fdr
        results_df.loc[i, "ghoshal_fdr"] = ghoshal_fdr
        results_df.loc[i, "peters_fdr"] = peters_fdr

        results_df.loc[i, "lingam_eid"] = lingam_eid
        results_df.loc[i, "dlingam_eid"] = dlingam_eid
        results_df.loc[i, "peters_eid"] = peters_eid
        results_df.loc[i, "ghoshal_eid"] = ghoshal_eid

        if fast_baselines:
            results_df.to_csv(
                f"baseline_results/df_d={dim}_{sem_type}_p={degree}_n={n_data}_ev={do_ev_noise}_n_seeds={n_seeds}_fast.csv"
            )
        else:
            print("Running bootstrap LiNGAM")
            (bootstrap_lingam_W, bootstrap_lingam_dict,) = run_bootstrapped_LiNGAM(
                Xs, test_Xs, ground_truth_W, do_ev_noise
            )
            # No need for this since we don't report it anywhere
            # print("Running BC_MCMC")
            # (
            #     bc_mcmc_out_stats,
            #     bc_mcmc_shd_c,
            #     bc_mcmc_shd,
            #     bc_mcmc_tpr,
            #     bc_mcmc_fpr,
            #     bc_mcmc_fdr,
            # ) = run_bc_mcmc(Xs, ground_truth_W, do_ev_noise)

            print("Running GADGET")
            (
                gadget_Ws,
                gadget_stat_dict,
                gadget_shd_c,
                gadget_shd,
                gadget_tpr,
                gadget_fpr,
                gadget_fdr,
                est_noises,
            ) = run_gadget(
                Xs, test_Xs, ground_truth_W, do_ev_noise, n_sumu_iters=n_sumu_iters
            )

            print("Computing intervention distances...")
            try:
                gadget_eid = ensemble_intervention_distance(
                    ground_truth_W,
                    gadget_Ws,
                    ground_truth_sigmas,
                    est_noises,
                    sem_type,
                )
            except:
                gadget_eid = np.nan

            # results_df.loc[i, "bc_mcmc_shd_c"] = bc_mcmc_shd_c

            results_df.loc[i, "bootstrap_lingam_shd_c"] = bootstrap_lingam_dict["shd_c"]
            results_df.loc[i, "gadget_shd_c"] = gadget_stat_dict["shd_c"]

            results_df.loc[i, "bootstrap_lingam_shd"] = bootstrap_lingam_dict["shd"]
            results_df.loc[i, "gadget_shd"] = gadget_stat_dict["shd"]

            results_df.loc[i, "bootstrap_lingam_tpr"] = bootstrap_lingam_dict["tpr"]
            results_df.loc[i, "gadget_tpr"] = gadget_stat_dict["tpr"]

            results_df.loc[i, "bootstrap_lingam_fpr"] = bootstrap_lingam_dict["fpr"]
            results_df.loc[i, "gadget_fpr"] = gadget_stat_dict["fpr"]

            results_df.loc[i, "bootstrap_lingam_fdr"] = bootstrap_lingam_dict["fdr"]
            results_df.loc[i, "gadget_fdr"] = gadget_stat_dict["fdr"]

            results_df.loc[i, "bootstrap_lingam_auc"] = bootstrap_lingam_dict["auroc"]
            results_df.loc[i, "gadget_auc"] = gadget_stat_dict["auroc"]

            results_df.loc[i, "dlingam_eid"] = dlingam_eid
            results_df.loc[i, "gadget_eid"] = gadget_eid
            results_df.to_csv(
                f"baseline_results/df_d={dim}_{sem_type}_p={degree}_n={n_data}_ev={do_ev_noise}_n_seeds={n_seeds}.csv"
            )
            print(results_df.mean())
            print(results_df.std())

    print(f"Took {time.time() - t0}s to run {len(seeds)} seeds on dim {dim}")
    print("means:")
    print(results_df.mean())
    print("stds:")
    print(results_df.std())
    print("----------------------------------------")
    return results_df
