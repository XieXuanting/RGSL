import numpy as np
import scipy.sparse as sp

def load_data(path, dataset):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(np.int32))

    order_and_label = np.genfromtxt("{}{}.content".format(path, dataset),usecols=(0,-1),
                                        dtype=np.dtype(np.string_))


    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(order_and_label[:, -1])


    # build graph
    idx = np.array(order_and_label[:, 0], dtype=np.string_)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.string_)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #features = normalize(features)




    features = np.array(features.todense())
    labels = np.where(labels)[1]
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.todense()


    return adj, features, labels


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot