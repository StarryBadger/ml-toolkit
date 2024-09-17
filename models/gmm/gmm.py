import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.special import logsumexp

class GMM:
    def __init__(self, k, iteration_lim=100, tolerance=1e-3, reg_covar=1e-6):
        self.k = k
        self.iteration_lim = iteration_lim
        self.tolerance = tolerance
        self.reg_covar = reg_covar
        self.weights = None
        self.means = None
        self.covariances = None

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.weights = np.ones(self.k) / self.k
        random_indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.means = X[random_indices]
        self.covariances = np.array([np.eye(self.n_features) for _ in range(self.k)])
        
        log_likelihood = -np.inf

        for _ in range(self.iteration_lim):
            # E-step
            log_resp = self._e_step(X)

            # M-step
            self._m_step(X, log_resp)

            # Compute log-likelihood
            new_log_likelihood = self.getLikelihood(X)

            # Check for convergence
            if np.isfinite(new_log_likelihood) and np.abs(new_log_likelihood - log_likelihood) < self.tolerance:
                break

            log_likelihood = new_log_likelihood

    def _e_step(self, X):
        log_resp = np.zeros((X.shape[0], self.k))

        for k in range(self.k):
            log_resp[:, k] = np.log(self.weights[k] + 1e-15) + self._log_multivariate_normal_density(X, self.means[k], self.covariances[k])

        log_resp -= logsumexp(log_resp, axis=1)[:, np.newaxis]
        return log_resp

    def _m_step(self, X, log_resp):
        n_samples, n_features = X.shape

        resp = np.exp(log_resp)
        resp_sum = resp.sum(axis=0) + 1e-15

        self.weights = resp_sum / n_samples
        self.means = np.dot(resp.T, X) / resp_sum[:, np.newaxis]

        for k in range(self.k):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(resp[:, k] * diff.T, diff) / resp_sum[k] + self.reg_covar * np.eye(n_features)

    def _log_multivariate_normal_density(self, X, mean, cov):
        n_features = X.shape[1]
        log_det = np.linalg.slogdet(cov)[1]
        inv_cov = np.linalg.inv(cov)
        diff = X - mean
        maha = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        return -0.5 * (n_features * np.log(2 * np.pi) + log_det + maha)

    def getParams(self):
        return {
            'weights': self.weights,
            'means': self.means,
            'covariances': self.covariances
        }

    def getMembership(self, X):
        return np.exp(self._e_step(X))

    def getLikelihood(self, X):
        log_likelihood = logsumexp([np.log(self.weights[k] + 1e-15) + self._log_multivariate_normal_density(X, self.means[k], self.covariances[k]) for k in range(self.k)], axis=0)
        return np.sum(log_likelihood)
    
    def getHardAssignments(self, X):
        membership_probabilities = self.getMembership(X)
        return np.argmax(membership_probabilities, axis=1)
    

    def aic(self):
        num_means = self.k * self.n_features
        num_covariances = self.k * (self.n_features * (self.n_features + 1)) // 2 
        num_weights = self.k - 1
        num_params= num_means + num_covariances + num_weights
        log_likelihood = self.getLikelihood(self.X)
        return 2 * num_params - 2 * log_likelihood
    
    def bic(self):
        num_means = self.k * self.n_features
        num_covariances = self.k * (self.n_features * (self.n_features + 1)) // 2 
        num_weights = self.k - 1
        num_params= num_means + num_covariances + num_weights
        log_likelihood = self.getLikelihood(self.X)
        return num_params * np.log(self.n_samples) - 2 * log_likelihood
    
    