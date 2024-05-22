import os, json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm
from src.denoising_toy_utils import *

# # Fix seeds for reproducibility
# fix_seeds()

config = {
    'name': 'run_1', # Name of the run
    'x0_estimation': 'sample', # 'mean' or 'sample'
    'reduced_ddim_steps': 0, # DDIM steps, 0 corresponds to direct mapping to x1 as proposed in manuscript, only relevant if x0_estimation == 'sample'
    'model_pred_mode': 'x0', # 'x0', 'eps', 'mu', output of network
    'c_data': 1.0, # given for completeness but should be kept at 1.
    'c_residual': 0.005, # manuscript uses 0.1 for mean estimation, 0.005 for sample estimate; 0 for vanilla (uninformed) diffusion model
    'c_ineq': 0.0, # here provided for experimentation
    'lambda_opt': 0.0, # here provided for experimentation
    'true_randomness': False, # enable to consider an uninformative prior (unit Gaussian) instead of data
    'dim': 2, # Dimension of the hypersphere to be learned (2 for unit circle)
    'n_steps': 100, # Number of denoising timesteps
    'use_dynamic_threshold': False, # Enable dynamic threshold during denoising process
    'train_num_steps': 400, # Number of training steps
    'batch_size': 128, # Batch size
    'no_samples': 1000, # Number of samples to generate
    'sample_freq': 10, # Frequency of sampling w.r.t. training iterations
    'tot_eval_steps': 11, # Number of steps to visualize
    'fix_axes': True, # Fix axes for visualization
    'save_output': True, # Save model output and sample estimate
    'create_gif': False, # Create GIFs of denoising process for visualization (takes some time)
    'wandb_track': True # Enable Weights & Biases tracking
}

# Derived config
config['use_ddim_x0'] = config['x0_estimation'] == 'sample'
output_save_dir = f'./trained_models/toy/{config["name"]}'
os.makedirs(output_save_dir, exist_ok=True)

# Data
data = sample_hypersphere(10**4, config['dim'])
dataset = torch.tensor(data).float().to(device)

# Define model and optimizer
model = ConditionalModel(config['dim'], config['n_steps']).to(device)
optimizer = optim.Adam(model.parameters(), lr=5.e-4)
diff_dict = create_diff_dict(config['n_steps'], device)

# Define functions for residual, inequality and optimization evaluation
class ResidualFunc(nn.Module):
    '''
    Simple residual given by the unit hypersphere. (Replace this with your own residual evaluation.)
    '''
    def forward(self, x):
        return torch.sum(x**2, dim=1) - 1.0

class InequalityFunc(nn.Module):
    '''
    Just a simple example for experimentation. (Replace this with your own inequality evaluation.)
    '''
    def __init__(self, threshold, mode='leq'):
        super().__init__()
        self.threshold = threshold
        self.mode = mode
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        density = torch.sum(torch.abs(x), dim=1)
        shift = density - self.threshold
        return self.relu(shift if self.mode == 'leq' else -shift), density

class OptimizationFunc(nn.Module):
    '''
    Just a simple example for experimentation. (Replace this with your own optimization evaluation.)
    '''
    def forward(self, x):
        return x[:,0]

residual_func = ResidualFunc()
ineq_func = InequalityFunc(threshold=1.0, mode='leq')
opt_func = OptimizationFunc()

# Training and evaluation loop
def log_metrics(log_fn, metrics, step):
    for key, value in metrics.items():
        log_fn({key: value}, step=step)

def evaluate_and_log(seq, config):
    residual = residual_func(seq[0][-1]).abs().mean().item()
    metrics = {'residual_samples': residual}

    if config['c_ineq'] > 0:
        ineq_density = ineq_func(seq[0][-1])[1].abs().mean().item()
        metrics['density_samples'] = ineq_density

    if config['lambda_opt'] > 0:
        opt = opt_func(seq[0][-1]).abs().mean().item()
        metrics['opt_samples'] = opt

    return metrics

if config['wandb_track']:
    import wandb
    wandb.init(project='pidm_toy', name=config['name'])
    log_fn = wandb.log
else:
    log_fn = noop

pbar = tqdm(range(config['train_num_steps'] + 1))
eval_steps = np.linspace(0, config['n_steps'], config['tot_eval_steps']).astype(int)

for t in pbar:
    permutation = torch.randperm(dataset.size()[0])
    for i in range(0, dataset.size()[0], config['batch_size']):
        indices = permutation[i:i + config['batch_size']]
        batch_x = dataset[indices]
        if config['true_randomness']:
            batch_x = torch.randn_like(batch_x)

        loss, data_loss, residual_loss, ineq_loss, opt_loss = model_estimation_loss(
            model, batch_x, config['n_steps'], diff_dict, model_pred_mode=config['model_pred_mode'],
            residual_func=residual_func, ineq_func=ineq_func, opt_func=opt_func,
            c_data=config['c_data'], c_residual=config['c_residual'], c_ineq=config['c_ineq'],
            lambda_opt=config['lambda_opt'], use_ddim_x0=config['use_ddim_x0'], reduced_ddim_steps=config['reduced_ddim_steps'])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pbar.set_description(f'training loss: {loss.item():.4f}')
        log_metrics(log_fn, {
            'loss': loss.item(),
            'loss_data': data_loss,
            'loss_residual': residual_loss
        }, step=t)
        if config['c_ineq'] > 0:
            log_fn({'loss_inequality': ineq_loss}, step=t)
        if config['lambda_opt'] > 0:
            log_fn({'loss_optimization': opt_loss}, step=t)

    if t % config['sample_freq'] == 0:
        shape = [config['no_samples'], config['dim']]
        seqs = p_sample_loop(
            model, shape, config['n_steps'], diff_dict,
            model_pred_mode=config['model_pred_mode'], save_output=config['save_output'],
            surpress_noise=True, use_dynamic_threshold=config['use_dynamic_threshold'],
            reduced_ddim_steps=config['reduced_ddim_steps'])
        
        metrics = evaluate_and_log(seqs, config)
        log_metrics(log_fn, metrics, step=t)

        fig, axs = plt.subplots(1, config['tot_eval_steps'], figsize=(3 * config['tot_eval_steps'] - 3, 3))
        labels = ['sample', 'model_output', 'x0_estimate']
        for seq_idx, seq in enumerate(seqs):
            if seq:
                for i_idx, i in enumerate(eval_steps):
                    cur_x = seq[i].detach().cpu()
                    if config['fix_axes'] and seq_idx == 0 and i_idx == 0:
                        x_lim, y_lim = (cur_x[:, 0].min(), cur_x[:, 0].max()), (cur_x[:, 1].min(), cur_x[:, 1].max())
                    axs[i_idx].set_xlim(x_lim)
                    axs[i_idx].set_ylim(y_lim)
                    axs[i_idx].scatter(cur_x[:, 0], cur_x[:, 1], s=10, label=labels[seq_idx])
                    axs[i_idx].set_title(f'$q(\\mathbf{{x}}_{{{config["n_steps"] - i}}})$')
                    if i_idx == 0:
                        axs[i_idx].legend()

                if seq_idx == 0:
                    os.makedirs(f'{output_save_dir}/csv', exist_ok=True)
                    np.savetxt(f'{output_save_dir}/csv/step_{t}_{labels[seq_idx]}.csv', seq[-1].detach().cpu(), delimiter=',')

                if config['create_gif']:
                    seq = torch.stack(seq, dim=0).detach().cpu().numpy()
                    array_to_gif(seq, f'{output_save_dir}/step_{t}_{labels[seq_idx]}.gif', x_lim=x_lim, y_lim=y_lim, label=labels[seq_idx])

        plt.savefig(f'{output_save_dir}/step_{t}.png')
        plt.close(fig)

save_model(model, config['name'], diff_dict, config['train_num_steps'], config['n_steps'], config['dim'], config['model_pred_mode'], residual_func, ineq_func, opt_func)
with open(f'{output_save_dir}/config.json', 'w') as f:
    json.dump(config, f)

if config['wandb_track']:
    wandb.finish()
