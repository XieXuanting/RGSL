import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
import warnings
from Adam import adam
from LoadData import load_data
from tqdm import tqdm
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # Load data
    A, X, gnd = load_data(path="./data/", dataset="Texas")
    A = A.A

    # Store some variables
    XtX = X.T.dot(X)
    XXt = X.dot(X.T)
    N = X.shape[0]
    k = len(np.unique(gnd))
    I = np.eye(N)
    I2 = np.eye(X.shape[1])
    if sp.issparse(X):
        X = X.todense()
    a = 1e-3
    beta = 1e-2



    # Normalize A
    A1 = A
    A = A + I
    D = np.sum(A,axis=1)
    D1 = np.power(D,-0.5)
    D1[np.isinf(D1)] = 0
    D1 = np.diagflat(D1)
    D[np.isinf(D)] = 0
    D = np.diagflat(D)
    A = D1.dot(A).dot(D1)

    # Get filter
    Ls = I - A
    Filter = 0.5 * Ls
    Nbr_sum = I
    Nbr = np.zeros((N,N))


    for i in range(4):
        Filter = Filter.dot(0.5 * Ls)
    X_bar = Filter.dot(X)
    XtX_bar = X_bar.T.dot(X_bar)
    XXt_bar = X_bar.dot(X_bar.T)
    num_data = X_bar.shape[0]

    print("Initial!")
    print('Begin!\n')
    acc_epoch = 0
    S_re = XXt_bar
    C = 0.5 * (np.fabs(S_re) + np.fabs(S_re.T))
    Nbr = C
    Nbr[Nbr > 1e-3] = 1
    Nbr[Nbr < 1e-3] = 0
    Nbr = C
    for m in range(num_data):
        Nbr[m][m] = 0
    n_Nbr = np.sum(Nbr, axis=1)


    # Break point
    trigger = 0
    cut_point = 0

    # Optimize
    cf = None
    loss_last = 1e16
    loss_list = []

    for m in range(num_data):
        S_re[m][m] = 0
    for epoch in tqdm(range(30)):
        if trigger >= 20:
            break

        grad = np.zeros((num_data, num_data))
        X_Xt_S = XXt_bar.dot(S_re)

        for i in range(num_data):
            k0 = np.exp(S_re[i]).sum() - np.exp(S_re[i][i])
            for j in range(i+1, num_data):
                F1 = ((beta+1)*(np.linalg.norm(X_bar[i,:] - X_bar[j,:])**2))/(np.linalg.norm(X_bar[i,:] - X_bar[j,:])+beta)
                if Nbr[i][j] != 0:
                    F2 = 0
                    F2 = F2 + (-Nbr[i][j] + n_Nbr[i]*np.exp(S_re[i][j]) / k0)
                else:
                    F2 = 0
                grad[i][j] = a * F2 + F1
                grad[j][i] = grad[i][j]
        loss_all_node = 0
        loss_view = 0
        for i in (range(num_data)):
            k0 = np.exp(S_re[i]).sum() - np.exp(S_re[i][i])
            loss_nbr = 0
            for j in (range(num_data)):
                loss_nbr = loss_nbr - (np.log(np.exp(S_re[i][j]) / k0))*Nbr[i][j]
                loss_view += S_re[i][j]*((beta+1)*(np.linalg.norm(X_bar[i,:] - X_bar[j,:])**2))/(np.linalg.norm(X_bar[i,:] - X_bar[j,:])+beta)
            loss_all_node = loss_all_node + loss_nbr

        loss_S_re = a * loss_all_node + loss_view
        if loss_S_re < loss_last:
            loss_last = loss_S_re
        if loss_S_re > loss_last:
            cut_point += 1
            if cut_point >= 2:
                break
        loss_list.append(loss_last)
        S_re, cf = adam(S_re, grad, cf)
        C = 0.5 * (np.fabs(S_re) + np.fabs(S_re.T))
        u, s, v = sp.linalg.svds(C, k=k, which='LM')

        # Clustering
        kmeans = KMeans(n_clusters=k, random_state=23).fit(u)
        predict_labels = kmeans.predict(u)
        re_ = clustering_metrics(gnd, predict_labels)
        ac, nm, f1 = re_.evaluationClusterModelFromLabel()
        if ac > acc_epoch:
            acc_epoch = ac
            nmi_epoch = nm
            f1_epoch = f1
            best_epoch = epoch
        else:
            trigger += 1
        Nbr = C
        Nbr[Nbr > 1e-3] = 1
        Nbr[Nbr < 1e-3] = 0
        for m in range(num_data):
            Nbr[m][m] = 0
        n_Nbr = np.sum(Nbr, axis=1)
    print("acc= {:>.6f}".format(acc_epoch),
          "nmi= {:>.6f}".format(nmi_epoch),
          "f1= {:>.6f}".format(f1_epoch))
