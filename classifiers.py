import numpy as np
import scipy as sp

class MahalanobisClassifier():
    def __init__(self, samples, labels):
        self.clusters={}
        for lbl in np.unique(labels):
            self.clusters[lbl] = samples.loc[labels == lbl, :]

    def mahalanobis(self, x, data, cov=None):
        """Compute the Mahalanobis Distance between each row of x and the data  
        x    : vector or matrix of data with, say, p columns.
        data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
        cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
        """
        x_minus_mu = x - np.mean(data)
        if not cov:
            cov = np.cov(data.values.T)
        inv_covmat = sp.linalg.inv(cov)
        left_term = np.dot(x_minus_mu, inv_covmat)
        mahal = np.dot(left_term, x_minus_mu.T)
        return mahal.diagonal()

    def predict_probability(self, unlabeled_samples):
        dists = np.array([])
	
        def dist2prob(D):
           row_sums = D.sum(axis=1)
           D_norm = (D / row_sums[:, np.newaxis])
           S = 1 - D_norm
           row_sums = S.sum(axis=1)
           S_norm = (S / row_sums[:, np.newaxis])		   
           return S_norm       
		   
	    #Distance of each sample from all clusters
        for lbl in self.clusters:
            tmp_dists=self.mahalanobis(unlabeled_samples, self.clusters[lbl])
            if len(dists)!=0:
                dists = np.column_stack((dists, tmp_dists))
            else:
                dists = tmp_dists

        return dist2prob(dists)
        

    def predict_class(self, unlabeled_sample, ind2label):
        return np.array([ind2label[np.argmax(row)] for row in self.predict_probability(unlabeled_sample)])
