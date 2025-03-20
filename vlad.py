
import pickle
from sklearn.linear_model import Ridge
import torch
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree

class VLAD:
    def __init__(self, n_clusters=100, n_batchsize=1000000, gmp=False, gamma=1000, powernorm=True, from_dir = None):
         self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=1, batch_size=n_batchsize, verbose=1, random_state=41)
         self.gmp = gmp
         self.gamma = gamma
         self.powernorm = powernorm
         self.n_clusters = n_clusters
         self.from_dir = from_dir
         self.n_workers = 1
         
         if self.from_dir:
            self.load(self.from_dir)
            self._train_impl(None)
            print(f'Loaded KMeans model from {self.from_dir}')

    def __str__(self):
        return f'VLAD with {self.n_clusters} from {self.from_dir}'
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.kmeans, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.kmeans = pickle.load(f)

    def assignments(self, kdt, descriptors, clusters):
         descriptors = np.nan_to_num(descriptors, nan=0)

         dists, indices = kdt.query(descriptors, k=1)

         assignment = np.zeros((len(descriptors), len(clusters)))
         assignment[np.arange(len(descriptors)), indices.flatten()] = 1

         return assignment

    def _train_impl(self, train_descs):
        if self.from_dir is None:
            self.kmeans.fit(train_descs)

        self.kdt = [KDTree(self.kmeans.cluster_centers_,
                        metric='euclidean')]
        self.cluster_centers = [self.kmeans.cluster_centers_]

    def encode(self, document):
        for centroids, kdt in zip(self.cluster_centers, self.kdt):
            return self.process_helper(document, centroids, kdt)


    def process_helper(self, descs, cluster_centers, kdt):
        #  if self.from_dir is not None:
        #      try:
        #          desc = doc.load_features_from_disk(self.from_dir)
        #      except Exception as e:
        #          #print(e)
        #          desc = torch.zeros((1,384))

        #  else:
        #      desc = doc.descriptors
         T, D = descs.shape

         # Precompute assignments for all descriptors
         a = self.assignments(kdt, descs, cluster_centers)

         # Initialize feature encoding vector
         f_enc = np.zeros((D * len(cluster_centers)), dtype=np.float32)

         # Pre-compute constants
         num_clusters = cluster_centers.shape[0]

         # Compute residuals for all clusters and descriptors
         all_residuals = descs[:, np.newaxis, :] - cluster_centers

         for k in range(num_clusters):
            mask = a[:, k] > 0  # boolean mask for descriptors assigned to cluster k
            nn = np.sum(mask)  # number of descriptors assigned to cluster k
            if nn == 0:
                continue

            res = all_residuals[mask, k, :]

            # e) generalized max pooling
            if self.gmp:
                clf = Ridge(alpha=self.gamma, fit_intercept=False, solver='sparse_cg', max_iter=500)
                clf.fit(res, np.ones((nn,)))
                f_enc[k * D:(k + 1) * D] = clf.coef_
            else:
                f_enc[k * D:(k + 1) * D] = np.sum(res, axis=0)

         # c) power normalization
         if self.powernorm:
             f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

         # l2 normalization
         f_enc = normalize(f_enc.reshape(1, -1), norm='l2')
         return f_enc
        #  # Add to document encoding
        #  if doc.encoding is None:
        #      doc.encoding = f_enc
        #  else:
        #      doc.encoding = np.concatenate([doc.encoding, f_enc], axis=1)

if __name__ == '__main__':
    descs = np.random.rand(1000, 64)
    test_doc = np.random.rand(100,64)

    vlad = VLAD(gmp=False)
    vlad._train_impl(descs)
    # print(vlad.process(test_doc).shape)
