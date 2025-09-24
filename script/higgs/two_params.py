import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2
from sklearn.model_selection import train_test_split

# For parameterization!
import click

# For modularization!
from lfibm.estimator.likelihood import LikelihoodEstimator
from lfibm.simulator.higgs import HiggsSimulator


@click.command()
@click.option('--n', help='Sample size', type=int, default=10_000)
@click.option('--config_dir', type=str, default='~/research/lfi-benchmark/assets/config.ini')
@click.option('--val_split', help='Validation split fraction', type=float, default=0.3)
def main(n, config_dir, val_split):
    config = configparser.ConfigParser()
    config_path = Path(config_dir).expanduser()
    config.read(config_path)
    DATA_DIR = config["DEFAULT"]["DataDir"]
    OUTPUT_DIR = f'{config["DEFAULT"]["OutputDir"]}/higgs/n{n}_val_split{val_split}'
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    simulator = HiggsSimulator(input_dir=f'{DATA_DIR}', default_observable='PRI_had_pt')

    # Data generation
    Y, params, samples = simulator.simulate(size=n)

    # Data type conversion
    Y, params, samples = Y.numpy(), params.numpy(), samples.numpy()

    print(f"\nGenerated {n} samples")
    print(f"Parameter shape: {params.shape}")
    print(f"Sample shape: {samples.shape}")
    print(f"Labels: {np.sum(Y)} true samples, {np.sum(1-Y)} permuted samples")

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

    # Example: Estimate likelihood ratios for first 10 validation samples
    print("\nExample likelihood ratio estimates for first 10 validation samples:")
    test_params = val_params[:10]
    test_samples = val_samples[:10] 
    test_labels = val_Y[:10]

    likelihood_ratios = estimator.estimate_likelihood_ratio(test_params, test_samples)
    
    for i in range(10):
        print(f"Sample {i}: True label={test_labels[i]}, "
              f"Likelihood ratio={likelihood_ratios[i]:.4f}")

    # Get raw likelihood ratios first
    raw_likelihood_ratios = estimator.estimate_likelihood_ratio(val_params, val_samples)
    
    # Apply safety checks before taking log
    safe_mask = (raw_likelihood_ratios > 1e-10) & np.isfinite(raw_likelihood_ratios)
    safe_ratios = raw_likelihood_ratios[safe_mask]
    safe_Y = val_Y[safe_mask]
    
    # Calculate -2 * log(LR)
    minus_2_log_ratios = -2 * np.log(safe_ratios)
    
    # Additional safety check for the result
    finite_mask = np.isfinite(minus_2_log_ratios) & (minus_2_log_ratios >= 0)
    val_ratios_clean = minus_2_log_ratios[finite_mask]
    val_Y_clean = safe_Y[finite_mask]

    # Only use TRUE samples for chi-square comparison
    true_sample_ratios = val_ratios_clean[val_Y_clean == 1]
    
    # Clip extreme values for better visualization
    val_ratios_clipped = np.clip(true_sample_ratios, 0, np.percentile(true_sample_ratios, 99))
    
    # Extend x-range to better show the chi-square distribution
    x_range = np.linspace(0, 20, 1000)

    # Ensure output directory exists
    Path(f"{OUTPUT_DIR}/higgs").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(val_ratios_clipped, bins=50, alpha=0.7, 
             label=f'True samples -2×log(LR) (n={len(val_ratios_clipped)})', density=True)
    plt.plot(x_range, chi2.pdf(x_range, df=param_dim), 'r--', 
             label=f'Chi-square (df={param_dim})', linewidth=2)
    plt.xlabel('-2 × Log(Likelihood Ratio)')
    plt.ylabel('Density')
    plt.xlim(0, 20)
    plt.legend()
    plt.title('Distribution of -2×Log(LR) - Higgs Simulation')
    plt.savefig(f"{OUTPUT_DIR}/distribution_of_likelihood_ratio.png", dpi=150)
    plt.close()

    # Plot confidence intervals for 10 random samples
    true_indices = np.where(val_Y == 1)[0]
    selected_indices = np.random.choice(true_indices, min(10, len(true_indices)), replace=False)

    plt.figure(figsize=(12, 8))

    for i, idx in enumerate(selected_indices):
        sample = val_samples[idx]
        true_param = val_params[idx]  # True parameter value
        
        # For Higgs, we need to determine what parameter we're estimating
        # This depends on your HiggsSimulator implementation
        # Assuming it's a 1D parameter for now
        if param_dim == 1:
            # Grid search over parameter values
            param_grid = np.linspace(true_param - 2, true_param + 2, 50)
            log_likelihood_ratios = []
            
            for param_test in param_grid:
                test_params = np.array([[param_test]])
                test_samples = sample.reshape(1, -1)
                lr = estimator.estimate_likelihood_ratio(test_params, test_samples)[0]
                log_lr = -2 * np.log(np.clip(lr, 1e-10, np.inf))
                log_likelihood_ratios.append(log_lr)
        else:
            # For multi-dimensional parameters, fix all but the first dimension
            param_grid = np.linspace(true_param[0] - 2, true_param[0] + 2, 50)
            log_likelihood_ratios = []
            
            for param_test in param_grid:
                test_params = true_param.copy()
                test_params[0] = param_test
                test_params = test_params.reshape(1, -1)
                test_samples = sample.reshape(1, -1)
                lr = estimator.estimate_likelihood_ratio(test_params, test_samples)[0]
                log_lr = -2 * np.log(np.clip(lr, 1e-10, np.inf))
                log_likelihood_ratios.append(log_lr)
        
        log_likelihood_ratios = np.array(log_likelihood_ratios)
        
        # Find 95% confidence interval (where -2*log(LR) < 3.84)
        in_ci = log_likelihood_ratios < 3.84
        
        if np.any(in_ci):
            ci_indices = np.where(in_ci)[0]
            ci_lower = param_grid[ci_indices[0]]
            ci_upper = param_grid[ci_indices[-1]]
            
            # Plot confidence interval as horizontal line
            y_pos = i + 0.5
            plt.plot([ci_lower, ci_upper], [y_pos, y_pos], 'b-', linewidth=3, alpha=0.7)
            plt.plot([ci_lower, ci_upper], [y_pos, y_pos], 'bo', markersize=4)
            
            # True value
            true_val = true_param if param_dim == 1 else true_param[0]
            plt.plot(true_val, y_pos, 'ro', markersize=6)
            
            # Add text with CI bounds
            plt.text(param_grid.max() + 0.1, y_pos, f'[{ci_lower:.2f}, {ci_upper:.2f}]', 
                    va='center', fontsize=8)

    plt.xlabel('Parameter Value')
    plt.ylabel('Sample Index')
    plt.title('95% Confidence Intervals for Higgs Parameter\n(Blue lines = CI, Red dots = True values)')
    plt.xlim(param_grid.min() - 0.5, param_grid.max() + 1.5)
    plt.ylim(0, 10.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confidence_intervals.png", dpi=150)
    plt.close()

    return

if __name__ == "__main__":
    main()
