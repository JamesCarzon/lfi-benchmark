from typing import Dict, Optional, Tuple, Any
import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from torch.distributions import Categorical, MultivariateNormal

# From src
from lfibm.utils.data_generation import BoxUniform
from lfibm.utils.parallel import tqdm_joblib

# From submodule
from hep_challenge.datasets import Data
from hep_challenge.systematics import systematics


class HiggsSimulator:
    """
    Higgs simulator for generating synthetic data without lf2i dependency.
    """

    POI_SPACE_BOUNDS = {'low': torch.tensor(0), 'high': torch.tensor(5)}
    NP_SPACE_BOUNDS = {
        'tes': {'low': torch.tensor(0.0), 'high': torch.tensor(1.5)},
        'jes': {'low': torch.tensor(0.9), 'high': torch.tensor(1.1)},
        'soft_met': {'low': torch.tensor(0.0), 'high': torch.tensor(5.0)},
        'ttbar_scale': {'low': torch.tensor(0.8), 'high': torch.tensor(1.2)},
        'diboson_scale': {'low': torch.tensor(0.0), 'high': torch.tensor(2.0)},
        'bkg_scale': {'low': torch.tensor(0.99), 'high': torch.tensor(1.01)},
    }
    NP_NAMES = ['tes', 'jes', 'soft_met', 'ttbar_scale', 'diboson_scale', 'bkg_scale']

    def __init__(
        self,
        input_dir: str,
        prior: Optional = None,
        proposal: Optional = None,
        seed: Optional[int] = None,
        prior_kwargs: Optional[Dict[str, Any]] = None,
        default_observable: str = 'PRI_had_pt',
        batch_size: int = 1000,
    ):
        """
        Initialize the Higgs simulator.
        
        Args:
            input_dir: Directory containing input data
            prior: Prior distribution (if None, uses default MultivariateNormal)
            proposal: Proposal distribution (if None, uses default BoxUniform)
            seed: Random seed for reproducibility
            prior_kwargs: Additional kwargs for prior distribution
            default_observable: Default observable to use for simulations
            batch_size: Batch size for simulations
        """
        self.poi_dim = 1
        self.nuisance_dim = 6
        self.data_dim = 1
        self.batch_size = batch_size
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Load and prepare data
        self._data = Data(input_dir=input_dir, test_size=0.01)
        self._data.load_test_set()
        self._datasets = self._data.get_test_set()

        # Split datasets into signal and background
        self._signal_set = self._datasets['htautau'].reset_index(drop=True)
        self._background_set = pd.concat([
            self._datasets['ttbar'],
            self._datasets['diboson'],
            self._datasets['ztautau']
        ]).reset_index(drop=True)
        self._full_set = pd.concat([self._signal_set, self._background_set]).reset_index(drop=True)

        # Calculate signal proportion when mu = 1
        self._signal_prop_mu_equals_one = self._signal_set['weights'].sum() / self._full_set['weights'].sum()
        self._mu_upper_bound = np.ceil(np.array(np.min([
            self.POI_SPACE_BOUNDS['high'].numpy(),
            1 / self._signal_prop_mu_equals_one
        ])))

        # Normalize weights to sum to 1.0 each
        self._signal_set['weights'] /= self._signal_set['weights'].sum()
        self._background_set['weights'] /= self._background_set['weights'].sum()

        # Setup distributions
        self.proposal = proposal if proposal is not None else BoxUniform(
            low=torch.tensor(
                [self.POI_SPACE_BOUNDS['low']] +\
                [self.NP_SPACE_BOUNDS[name]['low'] for name in self.NP_NAMES]
            ),
            high=torch.tensor(
                [self.POI_SPACE_BOUNDS['high']] +\
                [self.NP_SPACE_BOUNDS[name]['high'] for name in self.NP_NAMES[:1]] +\
                [self.NP_SPACE_BOUNDS[name]['low'] for name in self.NP_NAMES[1:]] # NOTE: 5/6 NPs are "turned off"
            )
        )
        self.prior = self.proposal
        
        self.default_observable = default_observable
        self.energy_for_thresholding = 22  # GeV

    def _sample_with_replacement(self, mu: float, size: int) -> pd.DataFrame:
        """
        Sample from the dataset with replacement for given signal strength mu.
        
        Args:
            mu: Signal strength parameter
            size: Number of samples to draw
            
        Returns:
            DataFrame with sampled data
        """
        assert mu <= self._mu_upper_bound, f"mu ({mu}) exceeds upper bound ({self._mu_upper_bound})"

        # Draw number of signal events from binomial
        num_signal_to_draw = np.random.binomial(n=size, p=mu * self._signal_prop_mu_equals_one)
        num_bkg_to_draw = size - num_signal_to_draw

        samples_to_concat = []
    
        if num_signal_to_draw > 0:
            signal_data = self._signal_set.sample(
                n=num_signal_to_draw, 
                replace=True, 
                weights=self._signal_set['weights']
            )
            samples_to_concat.append(signal_data)
            
        if num_bkg_to_draw > 0:
            background_data = self._background_set.sample(
                n=num_bkg_to_draw, 
                replace=True, 
                weights=self._background_set['weights']
            )
            samples_to_concat.append(background_data)
        
        # Handle edge cases
        if len(samples_to_concat) == 0:
            return self._signal_set.iloc[:0].copy()
        elif len(samples_to_concat) == 1:
            return samples_to_concat[0].reset_index(drop=True)
        else:
            return pd.concat(samples_to_concat, ignore_index=True)

    def _apply_systematics(self,
                          sample: pd.DataFrame,
                          tes: float = 1.0,
                          jes: float = 1.0,
                          soft_met: float = 0.0,
                          ttbar_scale: Optional[float] = None,
                          diboson_scale: Optional[float] = None,
                          bkg_scale: Optional[float] = None,
                          dopostprocess: bool = False) -> pd.DataFrame:
        """Apply systematic uncertainties to the sample."""
        return systematics(
            data_set=sample,
            tes=tes,
            jes=jes,
            soft_met=soft_met,
            ttbar_scale=ttbar_scale,
            diboson_scale=diboson_scale,
            bkg_scale=bkg_scale,
            dopostprocess=dopostprocess,
        )

    def likelihood(self, theta: torch.Tensor, observable: Optional[str] = None) -> torch.Tensor:
        """
        Evaluate likelihood for given parameters.
        
        Args:
            theta: Parameters tensor of shape (batch_size, poi_dim + nuisance_dim)
            observable: Observable to use (if None, uses default)
            
        Returns:
            Tensor of simulated observables
        """
        theta = theta.reshape(-1, self.poi_dim + self.nuisance_dim)
        pois = theta[:, 0].unsqueeze(1)
        nuisances = theta[:, 1:]

        if observable is None:
            observable = self.default_observable
        else:
            assert observable in self._full_set.columns, f"Observable {observable} not found in dataset columns."

        # Prepare data for parallel processing
        signal_data = self._signal_set.copy()
        background_data = self._background_set.copy()
        signal_weights = self._signal_set['weights']
        background_weights = self._background_set['weights']
        
        # Prepare parameter tuples for workers
        worker_params = [
            (mu.item(), nu, observable, self.batch_size,
            signal_data, background_data, signal_weights,
            background_weights, self._signal_prop_mu_equals_one, self._mu_upper_bound)
            for mu, nu in zip(pois, nuisances)
        ]

        with tqdm_joblib(tqdm(range(len(worker_params)), desc='Sampling likelihood')) as _:
            results = Parallel(n_jobs=-1)(
                delayed(eval_one_worker)(params) for params in worker_params
            )
        
        return torch.stack(results)

    def simulate(self,
                size: int,
                estimation_method: str = 'likelihood',
                observable: Optional[str] = None,
                p: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate data using the specified method.
        
        Args:
            size: Number of samples to generate
            estimation_method: Method to use ('likelihood', 'prediction', 'posterior')
            observable: Observable to simulate (if None, uses default)
            p: Probability of sampling from proposal distribution
            
        Returns:
            Tuple of (class labels, parameters, samples)
        """
        if observable is None:
            observable = self.default_observable
            
        if estimation_method == 'likelihood':
            # Sample class labels
            Y = Categorical(probs=torch.Tensor([1-p, p])).sample(sample_shape=(size,)).long()

            # Count samples needed from each distribution
            n_marginal = (Y == 0).sum().item()
            n_proposal = (Y == 1).sum().item()

            # Sample parameters
            params_proposal = self.proposal.sample(sample_shape=(n_proposal,)).reshape(-1, self.poi_dim + self.nuisance_dim) \
                if n_proposal > 0 else torch.empty(0, self.poi_dim + self.nuisance_dim)
            
            params_marginal = self.proposal.sample(sample_shape=(n_marginal,)).reshape(-1, self.poi_dim + self.nuisance_dim) \
                if n_marginal > 0 else torch.empty(0, self.poi_dim + self.nuisance_dim)
            
            # For BFF algorithm, marginal class uses different "true" params
            params_marginal_eval = self.prior.sample(sample_shape=(n_marginal,)).reshape(-1, self.poi_dim + self.nuisance_dim) \
                if n_marginal > 0 else torch.empty(0, self.poi_dim + self.nuisance_dim)

            # Clip parameters to bounds
            if n_marginal > 0:
                low = torch.cat([self.POI_SPACE_BOUNDS['low'].unsqueeze(0)] + 
                               [self.NP_SPACE_BOUNDS[name]['low'].unsqueeze(0) for name in self.NP_NAMES])
                high = torch.cat([self.POI_SPACE_BOUNDS['high'].unsqueeze(0)] + 
                                [self.NP_SPACE_BOUNDS[name]['high'].unsqueeze(0) for name in self.NP_NAMES])
                params_marginal = torch.max(torch.min(params_marginal, high), low)
                params_marginal_eval = torch.max(torch.min(params_marginal_eval, high), low)

            # Generate samples
            samples_marginal = self.likelihood(theta=params_marginal_eval, observable=observable) \
                if n_marginal > 0 else torch.empty(0, self.data_dim * self.batch_size)
            samples_proposal = self.likelihood(theta=params_proposal, observable=observable) \
                if n_proposal > 0 else torch.empty(0, self.data_dim * self.batch_size)

            # Combine results
            params = torch.empty(size, self.poi_dim + self.nuisance_dim)
            samples = torch.empty(size, self.data_dim * self.batch_size)

            params[Y == 0, :] = params_marginal
            params[Y == 1, :] = params_proposal
            samples[Y == 0, :] = samples_marginal
            samples[Y == 1, :] = samples_proposal

            return Y, params[:, :2], samples
            
        elif estimation_method in ['prediction', 'posterior']:
            raise NotImplementedError(f"Estimation method '{estimation_method}' not implemented")
        else:
            raise ValueError(f"Only ['likelihood', 'prediction', 'posterior'] are supported, got {estimation_method}")

    def sample_prior(self, size: int) -> torch.Tensor:
        """Sample from the prior distribution."""
        return self.prior.sample(sample_shape=(size,))

    def sample_proposal(self, size: int) -> torch.Tensor:
        """Sample from the proposal distribution."""
        return self.proposal.sample(sample_shape=(size,))

    def log_prob_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate log probability under prior."""
        return self.prior.log_prob(theta)

    def log_prob_proposal(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate log probability under proposal."""
        return self.proposal.log_prob(theta)


def eval_one_worker(params):
    """
    Worker function for parallel processing of likelihood evaluation.
    """
    import numpy as np
    import pandas as pd
    import torch
    from hep_challenge.systematics import systematics
    
    (mu, nu, observable, batch_size, 
     signal_data, background_data, signal_weights, 
     background_weights, signal_prop, mu_upper_bound) = params
    
    # Validate mu
    assert mu <= mu_upper_bound, f"mu ({mu}) exceeds upper bound ({mu_upper_bound})"
    
    # Sample with replacement
    num_signal_to_draw = np.random.binomial(n=batch_size, p=mu * signal_prop)
    num_bkg_to_draw = batch_size - num_signal_to_draw

    # Reset indices for clean sampling
    signal_data = signal_data.reset_index(drop=True)
    background_data = background_data.reset_index(drop=True)

    samples_to_concat = []
    
    if num_signal_to_draw > 0:
        signal_sample = signal_data.sample(
            n=num_signal_to_draw, 
            replace=True, 
            weights=signal_weights
        ).reset_index(drop=True)
        samples_to_concat.append(signal_sample)
        
    if num_bkg_to_draw > 0:
        background_sample = background_data.sample(
            n=num_bkg_to_draw, 
            replace=True, 
            weights=background_weights
        ).reset_index(drop=True)
        samples_to_concat.append(background_sample)

    # Handle edge cases
    if len(samples_to_concat) == 0:
        sample = signal_data.iloc[:0].copy()
    elif len(samples_to_concat) == 1:
        sample = samples_to_concat[0].copy()
    else:
        sample = pd.concat(samples_to_concat, ignore_index=True)
    
    # Apply systematic uncertainties
    sample = systematics(
        data_set=sample,
        tes=nu[0].item(),
        jes=nu[1].item(),
        soft_met=nu[2].item(),
        ttbar_scale=nu[3].item() if len(nu) > 3 and nu[3] is not None else None,
        diboson_scale=nu[4].item() if len(nu) > 4 and nu[4] is not None else None,
        bkg_scale=nu[5].item() if len(nu) > 5 and nu[5] is not None else None,
        dopostprocess=False,
    )
    
    return torch.tensor(sample[observable].values, dtype=torch.float32)