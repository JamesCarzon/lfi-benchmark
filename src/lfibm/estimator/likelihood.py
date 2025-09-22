import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings


class LikelihoodEstimator:
    """
    Likelihood estimator using the likelihood ratio trick with sklearn MLPClassifier.
    
    Trains a classifier to distinguish between true and permuted samples,
    then uses it to estimate likelihood ratios: r(x,θ) = s(x,θ) / (1 - s(x,θ))
    """
    
    def __init__(self, hidden_layer_sizes=(128, 64), max_iter=500, random_state=42):
        """
        Initialize the likelihood estimator.
        
        Args:
            hidden_layer_sizes (tuple): Hidden layer dimensions
            max_iter (int): Maximum iterations for training
            random_state (int): Random seed
        """
        self.classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10
        )
        
    def _prepare_features(self, params, samples):
        """Concatenate parameters and samples into feature vectors."""
        if params.ndim == 1:
            params = params.reshape(1, -1)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
            
        # Handle broadcasting
        if params.shape[0] == 1 and samples.shape[0] > 1:
            params = np.repeat(params, samples.shape[0], axis=0)
        elif samples.shape[0] == 1 and params.shape[0] > 1:
            samples = np.repeat(samples, params.shape[0], axis=0)
            
        return np.concatenate([params, samples], axis=1)
    
    def fit(self, params, samples, labels, verbose=True):
        """
        Train the classifier.
        
        Args:
            params (np.array): Parameter vectors [N, param_dim]
            samples (np.array): Sample vectors [N, sample_dim] 
            labels (np.array): Binary labels (1 for true, 0 for permuted) [N]
            verbose (bool): Print training info
        """
        X = self._prepare_features(params, samples)
        
        self.classifier.fit(X, labels)
        
        if verbose:
            # Get training accuracy on full dataset
            train_acc = accuracy_score(labels, self.classifier.predict(X))
            print(f'Training completed. Final accuracy: {train_acc:.4f}')
            print(f'Converged in {self.classifier.n_iter_} iterations')
    
    def estimate_likelihood_ratio(self, params, samples):
        """
        Estimate likelihood ratio p(sample | param) / p_ref(sample).
        
        Args:
            params (np.array): Parameter vector(s)
            samples (np.array): Sample vector(s)
            
        Returns:
            np.array: Estimated likelihood ratios
        """
        X = self._prepare_features(params, samples)
        
        # Get classifier probabilities for class 1 (true samples)
        s = self.classifier.predict_proba(X)[:, 1]
        
        # Avoid division by zero
        s = np.clip(s, 1e-5, 1 - 1e-5)
        
        # Likelihood ratio = s / (1 - s)
        return s / (1 - s)
    
    def estimate_likelihood(self, params, samples, reference_density=None):
        """
        Estimate p(sample | param).
        
        Args:
            params (np.array): Parameter vector(s)
            samples (np.array): Sample vector(s)  
            reference_density (callable or float): Reference density p_ref(sample)
                
        Returns:
            np.array: Estimated likelihood values
        """
        likelihood_ratios = self.estimate_likelihood_ratio(params, samples)
        
        if reference_density is None:
            warnings.warn("No reference density provided. Returning likelihood ratios.")
            return likelihood_ratios
        elif callable(reference_density):
            return likelihood_ratios * reference_density(samples)
        else:
            return likelihood_ratios * reference_density
