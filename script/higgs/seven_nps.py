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
    OUTPUT_DIR = config["DEFAULT"]["OutputDir"]
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

    # Plot likelihood ratio distribution using validation set only
    val_ratios = estimator.estimate_likelihood_ratio(val_params, val_samples)
    finite_mask = np.isfinite(val_ratios)
    val_ratios_clean = val_ratios[finite_mask]
    val_Y_clean = val_Y[finite_mask]

    # Clip extreme values for better visualization
    val_ratios_clipped = np.clip(val_ratios_clean, 0, np.percentile(val_ratios_clean, 99))
    x_range = np.linspace(0, 10, 1000)

    # Ensure output directory exists
    Path(f"{OUTPUT_DIR}/higgs").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(val_ratios_clipped[val_Y_clean == 1], bins=50, alpha=0.7, 
             label=f'True samples (n={np.sum(val_Y_clean == 1)})', density=True)
    plt.plot(x_range, chi2.pdf(x_range, df=param_dim-1), 'r--', 
             label=f'Chi-square (df={param_dim - 1})', linewidth=2)
    plt.xlabel('Likelihood Ratio')
    plt.ylabel('Density')
    plt.xlim(0, 10)
    plt.legend()
    plt.title('Distribution of Likelihood Ratios (Validation Set)')
    plt.savefig(f"{OUTPUT_DIR}/higgs/distribution_of_likelihood_ratio.png", dpi=150)
    plt.close()

    return

if __name__ == "__main__":
    main()
