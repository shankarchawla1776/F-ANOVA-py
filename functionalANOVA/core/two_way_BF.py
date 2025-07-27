import numpy as np
from scipy import stats
from scipy.stats import chi2, f
from scipy.linalg import inv
import pandas as pd

# from .one_way_BF import one_way_BF
from .utils import aflag_maker

def two_way_BF(self, method, data, contrast, c):

    bflag = self.SubgroupIndicator
    N = self.N
    aflag = aflag_maker(self.n_i)
    dim = self.n_domain_points

    aflag0 = np.unique(aflag)
    p = len(aflag0)

    bflag0 = np.unique(bflag)
    q = len(bflag0)


    gsize = np.zeros((p,q))
    yy = []

    p_cell = []

    ij = 0

    for i in range(p):
        q_cell = []

        for j in range(q):
            ij += 1
            ijflag = (aflag == aflag0[i]) & (bflag == bflag0[j])
            nij = np.sum(ijflag)
            gsize[i,j] = nij

            yij = data[ijflag,:]
            yy.append(np.column_stack([ij * np.ones(nij), yij]))
            q_cell.append(nij)
        p_cell.append(q_cell)

    self.n_ii = p_cell


    yy = np.vstack(yy)

    if self.Weights == "UNIFORM":
        u = np.ones(p) / p
        v = np.ones(q) / q

    elif self.Weights == "PROPORTIONAL":
        u = np.sum(gsize, axis=1) / N
        v = np.sum(gsize, axis=0) / N

    Ap = np.eye(p) - np.outer(np.ones(p), u)
    Aq = np.eye(q) - np.outer(np.ones(q), v)

    if contrast is None or len(contrast) == 0:
        Hp = np.hstack([np.eye(p-1), -np.ones((p-1,1))])
        Hq = np.hstack([np.eye(q-1), -np.ones((q-1,1))])

        if self.Hypothesis == "INTERACTION":
            H = np.kron(Hp, Hq)
        elif self.Hypothesis == "PRIMARY":
            H = Hp
        elif self.Hypothesis == "SECONDARY":
            H = Hq
    else:
        H = contrast

    if self.Hypothesis == "INTERACTION":
        contrast_final = H @ np.kron(Ap, Aq)
    elif self.Hypothesis == "PRIMARY":
        contrast_final = H @ np.kron(Ap, v.reshape(1, -1))
    elif self.Hypothesis == "SECONDARY":
        contrast_final = H @ np.kron(u.reshape(1, -1), Aq)
    else:
        contrast_final = contrast

    pure_data = yy[:,1:]
    A = yy[:,0]

    pvalue, stat = one_way_BF(method, pure_data, contrast_final, c, indicator_a=A)

    return pvalue, stat
