import logging
import numpy as np
import networkx as nx
from typing import Dict
import jax.random as rnd
import numpy as np
from typing import Any
from cdt.data import load_dataset

# from dag_data import is_dag
import csv

debug_list = lambda x: list(x)


class SyntheticDataset(object):
    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        n,
        d,
        graph_type,
        degree,
        sem_type,
        noise_scale=1.0,
        dataset_type="linear",
        quadratic_scale=None,
    ):
        self.n = n
        self.d = d
        self.graph_type = graph_type
        self.degree = degree
        self.sem_type = sem_type
        self.noise_scale = noise_scale
        self.dataset_type = dataset_type
        self.w_range = (0.5, 2.0)
        self.quadratic_scale = quadratic_scale

        self._setup()
        self._logger.debug("Finished setting up dataset class")

    def _setup(self):
        self.W, self.W_2, self.P = SyntheticDataset.simulate_random_dag(
            self.d,
            self.degree,
            self.graph_type,
            self.w_range,
            (self.dataset_type != "linear"),
        )
        if self.dataset_type != "linear":
            assert self.W_2 is not None
            self.W_2 = self.W_2 * self.quadratic_scale

        self.X = SyntheticDataset.simulate_sem(
            self.W,
            self.n,
            self.sem_type,
            self.w_range,
            self.noise_scale,
            self.dataset_type,
            self.W_2,
        )

    @staticmethod
    def simulate_random_dag(d, degree, graph_type, w_range, return_w_2=False):
        """Simulate random DAG with some expected degree.

        Args:
            d: number of nodes
            degree: expected node degree, in + out
            graph_type: {erdos-renyi, barabasi-albert, full}
            w_range: weight range +/- (low, high)
            return_w_2: boolean, whether to return an additional
                weight matrix used for quadratic terms

        Returns:
            W: weighted DAG
            [Optional] W: weighted DAG with same occupancy but different weights
        """
        if graph_type == "erdos-renyi":
            prob = float(degree) / (d - 1)
            B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
        elif graph_type == "barabasi-albert":
            m = int(round(degree / 2))
            B = np.zeros([d, d])
            bag = [0]
            for ii in range(1, d):
                dest = np.random.choice(bag, size=m)
                for jj in dest:
                    B[ii, jj] = 1
                bag.append(ii)
                bag.extend(dest)
        elif graph_type == "full":  # ignore degree, only for experimental use
            B = np.tril(np.ones([d, d]), k=-1)
        else:
            raise ValueError("Unknown graph type")
        # random permutation
        P = np.random.permutation(np.eye(d, d))  # permutes first axis only
        B_perm = P.T.dot(B).dot(P)
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U[np.random.rand(d, d) < 0.5] *= -1
        W = (B_perm != 0).astype(float) * U
        U_2 = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U_2[np.random.rand(d, d) < 0.5] *= -1
        W_2 = (B_perm != 0).astype(float) * U_2

        # At the moment the generative process is P.T @ lower @ P, we want
        # it to be P' @ upper @ P'.T.
        # We can return W.T, so we are saying W.T = P'.T @ lower @ P.
        # We can then return P.T, as we have
        # (P.T).T @ lower @ P.T = W.T

        if return_w_2:
            return W.T, W_2.T, P.T
        else:
            return W.T, None, P.T

    @staticmethod
    def simulate_gaussian_dag(d, degree, graph_type, w_std):
        """Simulate dense DAG adjacency matrix

        Args:
            d: number of nodes
            degree: expected node degree, in + out
            graph_type: {erdos-renyi, barabasi-albert, full}
            w_range: weight range +/- (low, high)
            return_w_2: boolean, whether to return an additional
                weight matrix used for quadratic terms

        Returns:
            W: weighted DAG
            [Optional] W: weighted DAG with same occupancy but different weights
        """
        lower_entries = np.random.normal(loc=0.0, scale=w_std, size=(d * (d - 1) // 2))
        L = np.zeros((d, d))
        # We want the ground-truth W.T to be generated from PLP^\top
        # This is since we encode W.T as PLP^\top in the approach.
        L[np.tril_indices(d, -1)] = lower_entries
        P = np.random.permutation(np.eye(d, d))  # permutes first axis only
        W = (P @ L @ P.T).T
        return W, None, P, L

    @staticmethod
    def simulate_sem(
        W,
        n,
        sem_type,
        w_range,
        noise_scale=1.0,
        dataset_type="nonlinear_1",
        W_2=None,
        sigmas=None,
    ) -> np.ndarray:
        """Simulate samples from SEM with specified type of noise.

        Args:
            W: weigthed DAG
            n: number of samples
            sem_type: {linear-gauss,linear-exp,linear-gumbel}
            noise_scale: scale parameter of noise distribution in linear SEM

        Returns:
            X: [n,d] sample matrix
        """

        G = nx.DiGraph(W)
        d = W.shape[0]
        X = np.zeros([n, d], dtype=np.float64)

        if sigmas is None:
            sigmas = np.ones((d,)) * noise_scale

        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            if dataset_type == "linear":
                eta = X[:, parents].dot(W[parents, j])
            elif dataset_type == "quadratic":
                eta = X[:, parents].dot(W[parents, j]) + (X[:, parents] ** 2).dot(
                    W_2[parents, j]
                )
            else:
                raise ValueError("Unknown dataset type")

            if sem_type == "linear-gauss":
                X[:, j] = eta + np.random.normal(scale=sigmas[j], size=n)
            elif sem_type == "linear-exp":
                X[:, j] = eta + np.random.exponential(scale=sigmas[j], size=n)
            elif sem_type == "linear-gumbel":
                X[:, j] = eta + np.random.gumbel(scale=sigmas[j], size=n)
            else:
                raise ValueError("Unknown sem type")

        return X

    @staticmethod
    def intervene_sem(
        W, n, sem_type, sigmas=None, idx_to_fix=None, value_to_fix=None,
    ):
        """Simulate samples from SEM with specified type of noise.

        Args:
            W: weigthed DAG
            n: number of samples
            sem_type: {linear-gauss,linear-exp,linear-gumbel}
            noise_scale: scale parameter of noise distribution in linear SEM

        Returns:
            X: [n,d] sample matrix
        """

        G = nx.DiGraph(W)
        d = W.shape[0]
        X = np.zeros([n, d])
        if len(sigmas) == 1:
            sigmas = np.ones(d) * sigmas

        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            if j == idx_to_fix:
                X[:, j] = value_to_fix
            else:
                eta = X[:, parents].dot(W[parents, j])
                if sem_type == "linear-gauss":
                    X[:, j] = eta + np.random.normal(scale=sigmas[j], size=n)
                elif sem_type == "linear-exp":
                    X[:, j] = eta + np.random.exponential(scale=sigmas[j], size=n)
                elif sem_type == "linear-gumbel":
                    X[:, j] = eta + np.random.gumbel(scale=sigmas[j], size=n)
                else:
                    raise ValueError("Unknown sem type")

        return X


def count_accuracy(W_true, W_est, W_und=None) -> Dict["str", float]:
    """
    Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        W_true: ground truth graph
        W_est: predicted graph
        W_und: predicted undirected edges in CPDAG, asymmetric

    Returns in dict:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    B_true = W_true != 0
    B = W_est != 0
    B_und = None if W_und is None else W_und
    d = B.shape[0]

    # linear index of nonzeros
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        pred_und = np.flatnonzero(B_und)
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)  # type: ignore
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)  # type: ignore
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    if B_und is not None:
        B_lower += np.tril(B_und + B_und.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    return {"fdr": fdr, "tpr": tpr, "fpr": fpr, "shd": shd, "pred_size": pred_size}


def dagify(W):
    """Successively removes edges with smallest absolute weights
    until the graph with weight matrix W is a DAG"""
    import numpy as onp

    # def is_dag(W):
    #     return onp.abs(np.trace(jax.scipy.linalg.expm(W * W)) - dim) < 0.001

    def is_dag(W):
        G = nx.DiGraph(np.array(np.abs(W).T > 0))
        return nx.is_directed_acyclic_graph(G)

    dim = W.shape[0]
    while not is_dag(W):
        tmp = onp.array(W.copy())
        tmp[tmp == 0.0] = onp.nan
        min_idx = onp.nanargmin(onp.abs(tmp))
        W = onp.array(W.flatten())
        W[min_idx] = 0.0
        W = W.reshape((dim, dim))
    return W


def process_sachs(
    center: bool = True,
    print_labels: bool = False,
    normalize=False,
    n_data=None,
    rng_key=None,
):
    data = []
    with open("./data/sachs_observational.csv") as csvfile:
        filereader = csv.reader(csvfile, delimiter=",")
        for i, row in enumerate(filereader):
            if i == 0:
                if print_labels:
                    print(row)
                continue
            data.append(np.array([float(x) for x in row]).reshape((1, -1)))
    if n_data is None:
        data_out = np.concatenate(data, axis=0)
    else:
        if rng_key is None:
            data_out = np.concatenate(data, axis=0)[:n_data]
        else:
            data_out = np.concatenate(data, axis=0)
            idxs = rnd.choice(rng_key, len(data_out), shape=(n_data,), replace=False)
            data_out = data_out[idxs]

    if center:
        if normalize:
            data_out = (data_out - np.mean(data_out, axis=0)) / np.std(data_out, axis=0)
        else:
            data_out = data_out - np.mean(data_out, axis=0)

    return data_out


def get_sachs_ground_truth():
    """Labels are ['praf', 'pmek', 'plcg', 'PIP2', 'PIP3', 'p44/42', 'pakts473',
    'PKA', 'PKC', 'P38', 'pjnk']."""
    W = np.load("./data/sachs_ground_truth.npy")
    return W
