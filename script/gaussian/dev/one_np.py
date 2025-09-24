import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2, multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
import os

# For parameterization!
import click



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


class GaussianSimulator:
    """
    Gaussian simulator for mean estimation with nuisance parameter.
    
    Parameters: (mu, nu) where mu is parameter of interest and nu is nuisance parameter
    Samples: Multivariate Gaussian with mean [mu, nu] and identity covariance
    """
    
    def __init__(self, sigma=1.0):
        """
        Initialize Gaussian simulator.
        
        Args:
            sigma (float): Standard deviation for both dimensions
        """
        self.sigma = sigma
        self.cov = np.eye(2) * (sigma ** 2)
    
    def simulate(self, size):
        """
        Simulate data for likelihood ratio estimation.
        
        Args:
            size (int): Number of samples to generate
            
        Returns:
            Y (np.array): Binary labels (1 for true, 0 for permuted)
            params (np.array): Parameter vectors [mu, nu] 
            samples (np.array): Sample vectors
        """
        # Generate random parameters
        # mu (parameter of interest) ~ Uniform(-2, 2)
        # nu (nuisance parameter) ~ Uniform(-1, 1)
        mu = np.random.uniform(-2, 2, size)
        nu = np.random.uniform(-1, 1, size)
        params = np.column_stack([mu, nu])
        
        # Generate samples from multivariate Gaussian
        samples = np.zeros((size, 2))
        for i in range(size):
            mean = [params[i, 0], params[i, 1]]  # [mu, nu]
            samples[i] = np.random.multivariate_normal(mean, self.cov)
        
        # Generate labels: half true (1), half permuted (0)
        Y = np.concatenate([
            np.ones(size // 2),
            np.zeros(size - size // 2)
        ])
        
        # For permuted samples, shuffle the parameter-sample correspondence
        permuted_indices = np.arange(size // 2, size)
        np.random.shuffle(permuted_indices)
        
        # Permute parameters for the second half
        params[size // 2:] = params[permuted_indices]
        
        # Shuffle everything to mix true and permuted samples
        shuffle_idx = np.random.permutation(size)
        Y = Y[shuffle_idx]
        params = params[shuffle_idx]
        samples = samples[shuffle_idx]
        
        return torch.from_numpy(Y), torch.from_numpy(params), torch.from_numpy(samples)
    
    def true_likelihood(self, params, samples):
        """
        Compute the true likelihood p(sample | param).
        
        Args:
            params (np.array): Parameter vectors [mu, nu]
            samples (np.array): Sample vectors
            
        Returns:
            np.array: True likelihood values
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
            
        likelihoods = np.zeros(params.shape[0])
        
        for i in range(params.shape[0]):
            mean = params[i]  # [mu, nu]
            rv = multivariate_normal(mean=mean, cov=self.cov)
            likelihoods[i] = rv.pdf(samples[i])
            
        return likelihoods


@click.command()
@click.option('--n', help='Sample size', type=int, default=10_000)
@click.option('--config_dir', type=str, default='~/research/lfi-benchmark/assets/config.ini')
@click.option('--val_split', help='Validation split fraction', type=float, default=0.3)
@click.option('--sigma', help='Standard deviation for Gaussian', type=float, default=1.0)
def main(n, config_dir, val_split, sigma):
    config = configparser.ConfigParser()
    config_path = Path(config_dir).expanduser()
    config.read(config_path)
    OUTPUT_DIR = f'{config["DEFAULT"]["OutputDir"]}/gaussian/n{n}_val_split{val_split}_sigma{sigma}'
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize Gaussian simulator
    simulator = GaussianSimulator(sigma=sigma)

    # Data generation
    Y, params, samples = simulator.simulate(size=n)

    # Data type conversion
    Y, params, samples = Y.numpy(), params.numpy(), samples.numpy()

    print(f"\nGenerated {n} samples")
    print(f"Parameter shape: {params.shape}")
    print(f"Sample shape: {samples.shape}")
    print(f"Labels: {np.sum(Y)} true samples, {np.sum(1-Y)} permuted samples")
    print(f"Parameter ranges: mu ∈ [{params[:, 0].min():.2f}, {params[:, 0].max():.2f}], "
          f"nu ∈ [{params[:, 1].min():.2f}, {params[:, 1].max():.2f}]")

    # Split into training and validation sets
    train_params, val_params, train_samples, val_samples, train_Y, val_Y = train_test_split(
        params, samples, Y, 
        test_size=val_split, 
        random_state=42, 
        stratify=Y
    )
    
    print(f"\nTraining set: {len(train_Y)} samples")
    print(f"Validation set: {len(val_Y)} samples")

    # Initialize likelihood estimator
    param_dim = params.shape[1] if params.ndim > 1 else 1
    sample_dim = samples.shape[1] if samples.ndim > 1 else 1

    estimator = LikelihoodEstimator(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=1000,
        random_state=42
    )

    # Train the classifier on training set only
    print("\nTraining likelihood ratio estimator...")
    estimator.fit(train_params, train_samples, train_Y)

    # Plot confidence intervals for 10 random samples
    true_indices = np.where(val_Y == 1)[0]
    selected_indices = np.random.choice(true_indices, min(10, len(true_indices)), replace=False)

    plt.figure(figsize=(12, 8))

    for i, idx in enumerate(selected_indices):
        sample = val_samples[idx]
        true_mu = val_params[idx, 0]  # True μ value
        true_nu = val_params[idx, 1]  # True ν value
        
        # Grid search over μ values (fix ν to true value)
        mu_grid = np.linspace(-3, 3, 50)
        log_likelihood_ratios = []
        
        for mu_test in mu_grid:
            test_params = np.array([[mu_test, true_nu]])
            test_samples = sample.reshape(1, -1)
            lr = estimator.estimate_likelihood_ratio(test_params, test_samples)[0]
            log_lr = -2 * np.log(np.clip(lr, 1e-10, np.inf))
            log_likelihood_ratios.append(log_lr)
        
        log_likelihood_ratios = np.array(log_likelihood_ratios)
        
        # Find 95% confidence interval (where -2*log(LR) < 3.84)
        in_ci = log_likelihood_ratios < 3.84
        
        if np.any(in_ci):
            ci_indices = np.where(in_ci)[0]
            ci_lower = mu_grid[ci_indices[0]]
            ci_upper = mu_grid[ci_indices[-1]]
            
            # Plot confidence interval as horizontal line
            y_pos = i + 0.5
            plt.plot([ci_lower, ci_upper], [y_pos, y_pos], 'b-', linewidth=3, alpha=0.7)
            plt.plot([ci_lower, ci_upper], [y_pos, y_pos], 'bo', markersize=4)
            plt.plot(true_mu, y_pos, 'ro', markersize=6)  # True value
            
            # Add text with CI bounds
            plt.text(3.2, y_pos, f'[{ci_lower:.2f}, {ci_upper:.2f}]', 
                    va='center', fontsize=8)

    plt.xlabel('μ (parameter of interest)')
    plt.ylabel('Sample Index')
    plt.title('95% Confidence Intervals for μ\n(Blue lines = CI, Red dots = True values)')
    plt.xlim(-3.5, 4.5)
    plt.ylim(0, 10.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confidence_intervals.png", dpi=150)
    plt.close()

    print(f"\nConfidence intervals plotted for {len(selected_indices)} samples")

    # Example: Estimate likelihood ratios for first 10 validation samples
    print("\nExample likelihood ratio estimates for first 10 validation samples:")
    test_params = val_params[:10]
    test_samples = val_samples[:10] 
    test_labels = val_Y[:10]

    likelihood_ratios = estimator.estimate_likelihood_ratio(test_params, test_samples)
    
    # Also compute true likelihood ratios for comparison
    true_likelihoods = simulator.true_likelihood(test_params, test_samples)
    
    for i in range(10):
        print(f"Sample {i}: True label={test_labels[i]}, "
              f"Estimated LR={likelihood_ratios[i]:.4f}, "
              f"True likelihood={true_likelihoods[i]:.4f}")

    # Get raw likelihood ratios first
    raw_likelihood_ratios = estimator.estimate_likelihood_ratio(val_params, val_samples)
    
    # Apply safety checks before taking log
    # Remove any non-positive or non-finite values
    safe_mask = (raw_likelihood_ratios > 1e-10) & np.isfinite(raw_likelihood_ratios)
    safe_ratios = raw_likelihood_ratios[safe_mask]
    safe_Y = val_Y[safe_mask]
    
    # Calculate -2 * log(LR)
    minus_2_log_ratios = -2 * np.log(safe_ratios)
    
    # Additional safety check for the result
    finite_mask = np.isfinite(minus_2_log_ratios) & (minus_2_log_ratios >= 0)
    val_ratios_clean = minus_2_log_ratios[finite_mask]
    val_Y_clean = safe_Y[finite_mask]

    # Only use TRUE samples for chi-square comparison (this is key!)
    true_sample_ratios = val_ratios_clean[val_Y_clean == 1]
    
    # Clip extreme values for better visualization
    val_ratios_clipped = np.clip(true_sample_ratios, 0, np.percentile(true_sample_ratios, 99))
    
    # Extend x-range to better show the chi-square distribution
    x_range = np.linspace(0, 20, 1000)

    plt.figure(figsize=(10, 6))
    plt.hist(val_ratios_clipped, bins=50, alpha=0.7, 
             label=f'True samples -2×log(LR) (n={len(val_ratios_clipped)})', density=True)
    plt.plot(x_range, chi2.pdf(x_range, df=param_dim), 'r--', 
             label=f'Chi-square (df={param_dim})', linewidth=2)
    plt.xlabel('-2 × Log(Likelihood Ratio)')
    plt.ylabel('Density')
    plt.xlim(0, 20)  # Extended range
    plt.legend()
    plt.title(f'Distribution of -2×Log(LR) - Gaussian Simulation (σ={sigma})')
    plt.savefig(f"{OUTPUT_DIR}/distribution_of_likelihood_ratio.png", dpi=150)
    plt.close()

    return

if __name__ == "__main__":
    main()
