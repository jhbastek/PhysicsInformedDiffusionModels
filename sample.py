import os, yaml, time
import matplotlib.pyplot as plt
import pandas as pd
import torch
from src.data_utils import *
from torch.utils.data import DataLoader
from src.denoising_utils import *
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy
from src.residuals_mechanics_K import ResidualsMechanics

# Specify the path to your directory containing the model folders (e.g., the PIDM with mean estimation for the Darcy flow)

# NOTE: use below to evaluate the Darcy flow model with mean estimation
directory_path = './trained_models/darcy/'
name = 'PIDM-ME'
load_model_step = 300000

# NOTE: use below to evaluate the topology optimization model (with sample estimation)
# directory_path = './trained_models/mechanics/'
# name = 'PIDM'
# load_model_step = 600000

no_samples = 3 # number of samples to generate
create_gif = True # create GIFs of denoising process for visualization (may take some time)
topopt_eval = True # evaluate topopt metrics for topopt / mechanics study
eval_test_sets = True # evaluate both test sets for topopt / mechanics study

test_batches = -1 # set to -1 for full evaluation of both test sets

load_path = directory_path + name
config = yaml.safe_load(Path(load_path, 'model', 'model.yaml').read_text())

# DDIM evaluation for x0, not needed for inference (activate only for visualization)
use_ddim_x0 = False
ddim_steps = 0

# gradient guidance scale as in Super-Resolution Paper
residual_grad_guidance = config['residual_grad_guidance']
# residual corrections (NOTE: can also be changed AFTER training since these only affect inference)
correction_mode = config['correction_mode'] # 'x0', 'xt', CoCoGen use xt
M_correction = config['M_correction']
N_correction = config['N_correction']

# diffusion model params
gov_eqs = config['gov_eqs']
if gov_eqs != 'darcy' and residual_grad_guidance:
    raise ValueError('Gradient guidance only implemented for Darcy equation.')
fd_acc = config['fd_acc']
diff_steps = config['diff_steps']
use_dynamic_threshold = False
self_condition = False
use_double = False

save_output = True
eval_residuals = True

data_paths = None
if gov_eqs == 'darcy':
    # [xi_1,xi_2] -> [p,K]
    input_dim = 2
    output_dim = 2
    pixels_at_boundary = True
    domain_length = 1.
    reverse_d1 = True
    bcs = 'none' # 'none', 'periodic'
    pixels_per_dim = 64
    return_optimizer = False
    return_inequality = False
    train_batch_size = 32
    if name == 'local_test':
        no_samples = 1
        train_batch_size = 8
    sigmoid_last_channel = False
elif gov_eqs == 'mechanics':
    input_dim = 2
    output_dim = 3
    # [xi_1,xi_2] -> [u_1,u_2,rho]
    pixels_at_boundary = True
    domain_length = 64.
    reverse_d1 = True
    data_paths_valid = ('./data/mechanics/test/valid/fields/')
    data_paths_test_level_1 = ('./data/mechanics/test/test_level_1/fields/')
    data_paths_test_level_2 = ('./data/mechanics/test/test_level_2/fields/')
    bcs = 'none' # 'none', 'periodic'
    pixels_per_dim = 64
    return_optimizer = True
    return_inequality = True
    ds_valid = Dataset_Paths(data_paths_valid, use_double=use_double)
    ds_test_level_1 = Dataset_Paths(data_paths_test_level_1, use_double=use_double)
    ds_test_level_2 = Dataset_Paths(data_paths_test_level_2, use_double=use_double)
    train_batch_size = 5
    dl_valid = cycle(DataLoader(ds_valid, batch_size = train_batch_size, shuffle=False))
    dl_test_level_1 = DataLoader(ds_test_level_1, batch_size = train_batch_size, shuffle=True)
    dl_test_level_2 = DataLoader(ds_test_level_2, batch_size = train_batch_size, shuffle=True)
    sigmoid_last_channel = True
else:
    raise ValueError('Unknown governing equations.')
    
output_save_dir = load_path + '/evaluation'
# check if output dir exists, if yes iterate name, e.g. evaluation_1, evaluation_2, ...
if os.path.exists(output_save_dir):
    i = 1
    while os.path.exists(output_save_dir + f'_{i}/'):
        i += 1
    output_save_dir += f'_{i}/'
os.makedirs(output_save_dir, exist_ok=True)

if use_double:
    torch.set_default_dtype(torch.float64)

diffusion_utils = DenoisingDiffusion(diff_steps, device, residual_grad_guidance)

if gov_eqs == 'darcy':
    model = Unet3D(dim = 32, channels = output_dim, sigmoid_last_channel = sigmoid_last_channel).to(device)
elif gov_eqs == 'mechanics':
    model = Unet3D(dim = 128, channels = output_dim+3+4, out_dim = output_dim, sigmoid_last_channel = sigmoid_last_channel).to(device) # since we besides xt also supply vf, strain_energy_dens, von-mises, and 2 BCs and 2 loads

load_model(Path(load_path, 'model', 'checkpoint_' + str(load_model_step) + '.pt'), model)

if gov_eqs == 'darcy':
    residuals = ResidualsDarcy(model = model, fd_acc = fd_acc, pixels_per_dim = pixels_per_dim, pixels_at_boundary = pixels_at_boundary, reverse_d1 = reverse_d1, device = device, bcs = bcs, domain_length = domain_length, residual_grad_guidance = residual_grad_guidance, use_ddim_x0 = use_ddim_x0, ddim_steps = ddim_steps)
elif gov_eqs == 'mechanics':
    residuals = ResidualsMechanics(model = model, pixels_per_dim = pixels_per_dim, pixels_at_boundary = pixels_at_boundary, device = device, bcs = bcs, no_BC_folder = './data/mechanics/solidspy_k_no_BC/', topopt_eval = topopt_eval, use_ddim_x0 = use_ddim_x0, ddim_steps = ddim_steps)
else:
    raise ValueError('Unknown residuals mode.')

# count number of trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_params}')

# generate sample based on validation set
if gov_eqs == 'darcy':
    conditioning_input = None
    sample_shape = (no_samples, output_dim, pixels_per_dim, pixels_per_dim)
elif gov_eqs == 'mechanics':
    cur_batch = next(dl_valid).to(device)
    if cur_batch.shape[0] < no_samples:
        no_samples = cur_batch.shape[0] # reduce no_samples to batch size
    sample_shape = (no_samples, output_dim, pixels_per_dim+1, pixels_per_dim+1)
    cur_batch = cur_batch[torch.randperm(cur_batch.shape[0], device = device)[:no_samples]]
    conditioning, x_0, bcs = torch.tensor_split(cur_batch, (3, 6), dim=1) # vf_arr, strain_energy_density_fem, von_mises_stress, disp_x, disp_y, E_field, BC_node_x, BC_node_y, load_x_img, load_y_img
    conditioning_input = (conditioning, bcs, x_0)

output = diffusion_utils.p_sample_loop(conditioning_input, sample_shape, 
                        save_output=save_output, surpress_noise=True, 
                        use_dynamic_threshold=use_dynamic_threshold, 
                        residual_func=residuals, eval_residuals = eval_residuals, 
                        return_optimizer = return_optimizer, return_inequality = return_inequality,
                        M_correction = M_correction, N_correction = N_correction, correction_mode = correction_mode)

if eval_residuals:
    seqs = output[0]
    residual = output[1]['residual']
    residual = residual.abs().mean(dim=tuple(range(1, residual.ndim))) # reduce to batch dim
    if return_optimizer:
        optimized_quant = output[1]['optimized_quant']
    if return_inequality:
        ineq = output[1]['inequality_quant']
else:
    seqs = output
    
output_save_dir_validation = output_save_dir + f'/validation/step_{load_model_step}/'
os.makedirs(output_save_dir_validation, exist_ok=True)

if gov_eqs == 'mechanics':
    # save conditioning data for later evaluation
    cond_data = torch.cat((conditioning, x_0, bcs), dim=1)
    for cur_sample in range(no_samples):
        for channel_idx in range(cond_data.shape[1]):
            os.makedirs(output_save_dir_validation + f'sample_{cur_sample}/', exist_ok=True)
            np.savetxt(output_save_dir_validation + f'sample_{cur_sample}/cond_channel_{channel_idx}.csv', cond_data[cur_sample, channel_idx].detach().cpu().numpy(), delimiter=',')
        
labels = ['sample', 'model_output']
for seq_idx, seq in enumerate(seqs):

    # NOTE: We here only evaluate the sample at the final timestep and skip model_output as this is identical (since no noise is applied in last step).
    if seq_idx == 1:
        continue

    # remove frame dimension
    seq = torch.stack(seq, dim=0)
    if len(seq.shape) == 6:
        seq = seq.squeeze(-3)
        
    last_preds = seq[-1].numpy()
    sel_samples = np.arange(no_samples)
    channels = np.arange(output_dim)

    for sel_sample in sel_samples:
        for sel_channel in channels:
            last_pred = last_preds[sel_sample, sel_channel]
            last_pred_normalized = (last_pred - last_pred.min()) / (last_pred.max() - last_pred.min()) # normalize to [0,1]
            image = np.uint8(last_pred_normalized * 255)
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            if eval_residuals:
                title = f'residual: {residual[sel_sample]:.2e}'
                if return_optimizer:
                    title += f'\nopt: {optimized_quant[sel_sample]:.2f}'
                if return_inequality:
                    title += f'\nineq: {ineq[sel_sample]:.2e}'
                plt.title(title, color='green')
            filename = labels[seq_idx] + '_sample_' + str(sel_sample) + '_' + str(sel_channel) + '.png'
            plt.savefig(output_save_dir_validation + filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            os.makedirs(output_save_dir_validation + f'/sample_{sel_sample}/', exist_ok=True)
            np.savetxt(output_save_dir_validation + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.csv', last_pred, delimiter=',')

            if create_gif:
                sel_seq = seq[:, sel_sample, sel_channel].detach().cpu().numpy()
                image_array_to_gif(sel_seq, output_save_dir_validation + f'/sample_{sel_sample}/' + labels[seq_idx]  + '_' + str(sel_channel) + '.gif')

if eval_residuals:
    residuals_array = residual.detach().cpu().numpy()
    ineq_array = ineq.detach().cpu().numpy() if return_inequality else None
    optimized_quant_array = optimized_quant.detach().cpu().numpy() if return_optimizer else None

    df_data = {'Sample Index': list(range(no_samples)) + ['Mean'],
            'Residuals (abs)': list(residuals_array)}

    # add optimized quant data if available
    if return_optimizer:
        df_data['Optimized quantity'] = list(optimized_quant_array)
    
    # add inequality data if available
    if return_inequality:
        df_data['Inequality'] = list(ineq_array)

    # compute and append mean values
    df_data['Residuals (abs)'].append(np.nanmean(residuals_array))
    if return_optimizer:
        df_data['Optimized quantity'].append(np.nanmean(optimized_quant_array))
    if return_inequality:
        df_data['Inequality'].append(np.nanmean(ineq_array))
        
    df = pd.DataFrame(df_data)
    csv_path = os.path.join(output_save_dir_validation, 'sample_statistics.csv')
    df.to_csv(csv_path, index=False)

# do a full evaluation on both test sets for topopt study
with torch.no_grad():
    start_time = time.time()
    if eval_test_sets and gov_eqs == 'mechanics':
        test_datasets = [dl_test_level_1, dl_test_level_2]
        test_datasets_names = ['test_level_1', 'test_level_2']
        for ds_test_idx, dl_test in enumerate(test_datasets):
            residual_mean_abs_list, rel_CE_error_list, rel_vf_error_list, fm_error_list = [], [], [], []
            for batch_idx, batch in enumerate(dl_test):
                cur_batch = batch.to(device)
                sample_shape = (cur_batch.shape[0], output_dim, pixels_per_dim+1, pixels_per_dim+1)
                conditioning, x_0, bcs = torch.tensor_split(cur_batch, (3, 6), dim=1) # vf_arr, strain_energy_density_fem, von_mises_stress, disp_x, disp_y, E_field, BC_node_x, BC_node_y, load_x_img, load_y_img
                conditioning_input = (conditioning, bcs, x_0)
                output = diffusion_utils.p_sample_loop(conditioning_input, sample_shape, 
                                        save_output=save_output, surpress_noise=True,
                                        use_dynamic_threshold=use_dynamic_threshold,
                                        residual_func=residuals, eval_residuals = eval_residuals,
                                        return_optimizer = return_optimizer, return_inequality = return_inequality,
                                        M_correction = M_correction, N_correction = N_correction, correction_mode = correction_mode)
                if eval_residuals:
                    seqs = output[0]
                    residual = output[1]['residual']
                    residual = residual.abs().mean(dim=tuple(range(1, residual.ndim))) # reduce to batch dim
                    if return_optimizer:
                        optimized_quant = output[1]['optimized_quant']
                    if return_inequality:
                        ineq = output[1]['inequality_quant']
                else:
                    seqs = output                
                output_save_dir_tests = output_save_dir + '/' + test_datasets_names[ds_test_idx] + '/'
                os.makedirs(output_save_dir_tests, exist_ok=True)
                if batch_idx == 0: # only store first batch fully
                    labels = ['sample', 'model_output']
                    # NOTE: We here only evaluate the sample at the final timestep and skip model_output as this is identical (since no noise is applied in last step).
                    for seq_idx, seq in enumerate(seqs):
                        if seq_idx == 1:
                            continue
                        # remove frame dimension
                        seq = torch.stack(seq, dim=0)
                        if len(seq.shape) == 6:
                            seq = seq.squeeze(-3)                            
                        last_preds = seq[-1].numpy()
                        sel_samples = np.arange(len(last_preds))
                        channels = np.arange(output_dim)
                        for sel_sample in sel_samples:
                            # save conditioning data for later evaluation
                            cond_data = torch.cat((conditioning, x_0, bcs), dim=1)[sel_sample]
                            for channel_idx in range(cond_data.shape[0]):
                                os.makedirs(output_save_dir_tests + f'/sample_{sel_sample}/', exist_ok=True)
                                np.savetxt(output_save_dir_tests + f'/sample_{sel_sample}/cond_channel_{channel_idx}.csv', cond_data[channel_idx].detach().cpu().numpy(), delimiter=',')
                            for sel_channel in channels:
                                last_pred = last_preds[sel_sample, sel_channel]
                                last_pred_normalized = (last_pred - last_pred.min()) / (last_pred.max() - last_pred.min()) # normalize to [0,1]
                                image = np.uint8(last_pred_normalized * 255)
                                fig, ax = plt.subplots()
                                ax.imshow(image, cmap='gray', vmin=0, vmax=255)
                                ax.axis('off')
                                if eval_residuals:
                                    title = f'eq: {residual[sel_sample]:.2e}'
                                    if return_optimizer:
                                        title += f'\nopt: {optimized_quant[sel_sample]:.2f}'
                                    if return_inequality:
                                        title += f'\nineq: {ineq[sel_sample]:.2e}'
                                    plt.title(title, color='green')
                                filename = labels[seq_idx] + '_sample_' + str(sel_sample) + '_' + str(sel_channel) + '.png'
                                plt.savefig(output_save_dir_tests + filename, bbox_inches='tight', pad_inches=0)
                                plt.close(fig)
                                os.makedirs(output_save_dir_tests + f'/sample_{sel_sample}/', exist_ok=True)
                                np.savetxt(output_save_dir_tests + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.csv', last_pred, delimiter=',')
                                if create_gif:
                                    sel_seq = seq[:, sel_sample, sel_channel].detach().cpu().numpy()
                                    image_array_to_gif(sel_seq, output_save_dir_tests + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.gif')

                if eval_residuals:
                    residuals_array = residual.detach().cpu().numpy()
                    residual_mean_abs_list.append(residuals_array)
                if topopt_eval:
                    rel_CE_error = output[1]['rel_CE_error_full_batch'].detach().cpu().numpy()
                    rel_vf_error = output[1]['vf_error_full_batch'].detach().cpu().numpy()
                    fm_error = output[1]['fm_error_full_batch'].detach().cpu().numpy()
                    rel_CE_error_list.append(rel_CE_error)
                    rel_vf_error_list.append(rel_vf_error)
                    fm_error_list.append(fm_error)

                if test_batches != -1 and batch_idx > test_batches:
                    break

            if eval_residuals:
                residuals_array = np.concatenate(residual_mean_abs_list, axis=0)
                np.savetxt(output_save_dir_tests + 'residuals.csv', residuals_array, delimiter=',')
            if topopt_eval:
                rel_CE_error = np.concatenate(rel_CE_error_list, axis=0)
                rel_vf_error = np.concatenate(rel_vf_error_list, axis=0)
                fm_error = np.concatenate(fm_error_list, axis=0)
                np.savetxt(output_save_dir_tests + 'rel_CE_error.csv', rel_CE_error, delimiter=',')
                np.savetxt(output_save_dir_tests + 'rel_vf_error.csv', rel_vf_error, delimiter=',')
                np.savetxt(output_save_dir_tests + 'fm_error.csv', fm_error, delimiter=',')

            print(f'Evaluation of {name}: \n On {test_datasets_names[ds_test_idx]}.')
            print('CE median error:', np.median(rel_CE_error), 'VF mean error:', np.mean(rel_vf_error), 'FM mean error:', np.mean(fm_error), 'Mean residual:', np.mean(residuals_array), 'Median residual:', np.median(residuals_array))

    end_time = time.time()
    print(f'Evaluation for model {name} done (time: {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}).')