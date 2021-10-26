import mine
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

# a, b, c = 4, 3, 9
# array_1 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.intc)
# array_2 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.intc)
n_batch = 256
dim = 16
array_1 = np.random.uniform(0, 10, size=(n_batch, dim, dim))
t0 = time.time()
out = mine.compute(array_1)
print(f"Took {time.time() - t0}s for c version")
# print(out)


def multiple_linear_assignment(X):
    # X has shape (batch_size, k, k)
    X_out = np.zeros(X.shape[:-1], dtype=int)
    for i, x in enumerate(X):
        X_out[i] = linear_sum_assignment(x)[1]
    return X_out


t0 = time.time()
hung_out = multiple_linear_assignment(array_1)
print(f"Took {time.time() - t0}s for numpy version")
# print(hung_out)

assert np.all(out == hung_out)

t0 = time.time()
par_out = mine.compute_parallel(array_1)
assert np.all(par_out == hung_out)
print(f"Took {time.time() - t0}s for parallel c version")

print("All correct")
