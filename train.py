import os
# Removed gc import - manual garbage collection is inefficient
import torch
import torchaudio
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pesq import pesq
from scipy import interpolate
from torch.utils.data import DataLoader

from dataset_utils import SpeechDataset,subsample2,subsample4
from DCUnet10_TSTM.DCUnet import DCUnet10,DCUnet10_rTSTM,DCUnet10_cTSTM
from metrics import AudioMetrics2, AudioMetrics
from loss import RegularizedLoss

# First checking if GPU is available
train_on_gpu = torch.cuda.is_available()
if (train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
np.random.seed(999)
torch.manual_seed(999)

# If running on Cuda set these 2 for determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Audio backend is now automatically selected based on available dependencies

###################################### Parameters of Speech processing ##################################
SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

######################################## Datasets setting #########################################
# Choose white noise or different noise types in urbansound8K
noise_class = "white"

# Load white noise
if noise_class == "white":
    TRAIN_INPUT_DIR = Path('Datasets/WhiteNoise_Train_Input')
    TRAIN_TARGET_DIR = Path('Datasets/WhiteNoise_Train_Output')

    TEST_NOISY_DIR = Path('Datasets/WhiteNoise_Test_Input')
    TEST_CLEAN_DIR = Path('Datasets/clean_testset_wav')

# Load urbansound8K noise
else:
    TRAIN_INPUT_DIR = Path('Datasets/US_Class' + str(noise_class) + '_Train_Input')
    TRAIN_TARGET_DIR = Path('Datasets/US_Class' + str(noise_class) + '_Train_Output')

    TEST_NOISY_DIR = Path('Datasets/US_Class' + str(noise_class) + '_Test_Input')
    TEST_CLEAN_DIR = Path('Datasets/clean_testset_wav')

train_input_files = sorted(list(TRAIN_INPUT_DIR.rglob('*.wav')))
train_target_files = sorted(list(TRAIN_TARGET_DIR.rglob('*.wav')))

test_noisy_files = sorted(list(TEST_NOISY_DIR.rglob('*.wav')))
test_clean_files = sorted(list(TEST_CLEAN_DIR.rglob('*.wav')))

print("No. of Training files:",len(train_input_files))
print("No. of Testing files:",len(test_noisy_files))

basepath = str(noise_class)
fixedpath = 'SNA-DF/DCUnet10_complex_TSTM_subsample2/'

os.makedirs(fixedpath + basepath,exist_ok=True)
os.makedirs(fixedpath + basepath+"/Weights",exist_ok=True)
respath = fixedpath + basepath + '/results.txt'
#os.makedirs(basepath+"/Samples",exist_ok=True)

######################################## Metrics for evaluation #########################################
def resample(original, old_rate, new_rate):
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old = np.linspace(0, duration, original.shape[0])
        time_new = np.linspace(0, duration, int(original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original

wonky_samples = []

def getMetricsonLoader(loader, net, use_net=True):
    """Optimized metrics evaluation with better memory management and error handling."""
    net.eval()
    
    metric_names = ["PESQ-WB", "PESQ-NB", "SNR", "SSNR", "STOI"]
    overall_metrics = [[] for _ in range(5)]
    processed_samples = 0
    skipped_samples = 0
    
    # Pre-allocate window tensor for efficiency
    window_tensor = torch.hann_window(N_FFT)
    
    try:
        with torch.no_grad():
            for i, data in enumerate(loader):
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} samples", end="\n")
                elif i % 10 == 0:
                    print(f"Processing sample {i + 1}", end="")
                else:
                    print(".", end="")
                    
                if i in wonky_samples:
                    print(f"\nSkipping problematic sample {i}")
                    skipped_samples += 1
                    continue
                    
                try:
                    x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft = data
                    
                    # Validate data
                    if any(tensor.numel() == 0 for tensor in [x_noisy_stft, x_clean_stft]):
                        print(f"\nSkipping empty sample {i}")
                        skipped_samples += 1
                        continue
                    
                    # Move to device efficiently
                    device = DEVICE
                    x_noisy_stft = x_noisy_stft.to(device, non_blocking=True)
                    x_clean_stft = x_clean_stft.to(device, non_blocking=True)
                    window_tensor = window_tensor.to(device)

                    if use_net:
                        # Forward pass through network
                        x_est = net(x_noisy_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, is_istft=True)
                        if x_est.numel() == 0:
                            print(f"\nNetwork produced empty output for sample {i}")
                            skipped_samples += 1
                            continue
                        x_est_np = x_est.view(-1).detach().cpu().numpy()
                    else:
                        # Use noisy signal directly 
                        x_est_np = x_noisy_stft.view(-1).detach().cpu().numpy()
                    
                    # Efficient ISTFT conversion for clean signal
                    x_clean_stft_squeezed = torch.squeeze(x_clean_stft, 1)
                    if x_clean_stft_squeezed.shape[-1] != 2:
                        raise ValueError(f"Expected real/imag format, got shape {x_clean_stft_squeezed.shape}")
                        
                    x_clean_complex = torch.complex(x_clean_stft_squeezed[..., 0], 
                                                   x_clean_stft_squeezed[..., 1])
                    x_clean_np = torch.istft(
                        x_clean_complex, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        normalized=True, 
                        window=window_tensor
                    ).view(-1).detach().cpu().numpy()

                    # Validate audio lengths match
                    min_len = min(len(x_clean_np), len(x_est_np))
                    if min_len == 0:
                        print(f"\nZero-length audio in sample {i}")
                        skipped_samples += 1
                        continue
                        
                    x_clean_np = x_clean_np[:min_len]
                    x_est_np = x_est_np[:min_len]

                    # Compute metrics with error handling
                    try:
                        metrics = AudioMetrics2(x_clean_np, x_est_np, 48000)
                        
                        # Resample for PESQ (with bounds checking)
                        ref_wb = resample(x_clean_np, 48000, 16000)
                        deg_wb = resample(x_est_np, 48000, 16000) 
                        
                        ref_nb = resample(x_clean_np, 48000, 8000)
                        deg_nb = resample(x_est_np, 48000, 8000)
                        
                        # Validate resampled lengths
                        if len(ref_wb) != len(deg_wb) or len(ref_nb) != len(deg_nb):
                            print(f"\nLength mismatch in resampled audio for sample {i}")
                            skipped_samples += 1
                            continue
                        
                        pesq_wb = pesq(16000, ref_wb, deg_wb, 'wb')
                        pesq_nb = pesq(8000, ref_nb, deg_nb, 'nb')
                        
                        # Validate metric values
                        metric_values = [pesq_wb, pesq_nb, metrics.SNR, metrics.SSNR, metrics.STOI]
                        if any(not np.isfinite(val) for val in metric_values):
                            print(f"\nNon-finite metric values in sample {i}: {metric_values}")
                            skipped_samples += 1
                            continue
                        
                        # Store valid metrics
                        for j, val in enumerate(metric_values):
                            overall_metrics[j].append(val)
                            
                        processed_samples += 1
                        
                    except Exception as metric_error:
                        print(f"\nMetric computation error for sample {i}: {str(metric_error)}")
                        skipped_samples += 1
                        continue
                        
                except Exception as sample_error:
                    print(f"\nError processing sample {i}: {str(sample_error)}")
                    skipped_samples += 1
                    continue
                    
        print(f"\nSample metrics computed: {processed_samples} processed, {skipped_samples} skipped")
        
        if processed_samples == 0:
            print("Warning: No valid samples processed!")
            return {metric: {"Mean": 0.0, "STD": 0.0, "Min": 0.0, "Max": 0.0} 
                   for metric in metric_names}
        
        # Compute statistics
        results = {}
        for i, metric_name in enumerate(metric_names):
            if overall_metrics[i]:  # Check if list is not empty
                values = np.array(overall_metrics[i])
                results[metric_name] = {
                    "Mean": float(np.mean(values)),
                    "STD": float(np.std(values)),
                    "Min": float(np.min(values)), 
                    "Max": float(np.max(values))
                }
            else:
                results[metric_name] = {"Mean": 0.0, "STD": 0.0, "Min": 0.0, "Max": 0.0}
                
        print("Statistics computed")
        
        addon = "(cleaned by model)" if use_net else "(pre denoising)"
        print(f"Metrics on test data {addon}")
        for i, metric_name in enumerate(metric_names):
            if overall_metrics[i]:
                mean_val = results[metric_name]["Mean"]
                std_val = results[metric_name]["STD"]
                print(f"{metric_name}: {mean_val:.3f}+/-{std_val:.3f}")
            else:
                print(f"{metric_name}: No valid samples")
                
        return results
        
    except Exception as e:
        print(f"\nCritical error in metrics evaluation: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
######################################## TRAIN #########################################

def train_epoch(net, train_loader, loss_fn, optimizer, gradient_accumulation_steps=1):
    """Optimized training epoch with gradient accumulation and better memory management."""
    net.train()
    train_ep_loss = 0.
    counter = 0
    accumulated_loss = 0.
    
    # Validate inputs
    if gradient_accumulation_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}")

    try:
        for batch_idx, (x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft) in enumerate(train_loader):
            # Validate batch data
            if any(tensor.numel() == 0 for tensor in [x_noisy_stft, g1_stft, g1_wav, g2_wav]):
                print(f"Warning: Empty tensors in batch {batch_idx}, skipping...")
                continue
                
            # Move tensors to device efficiently (batch move)
            try:
                g1_stft = g1_stft.to(DEVICE, non_blocking=True)
                g1_wav = g1_wav.to(DEVICE, non_blocking=True) 
                g2_wav = g2_wav.to(DEVICE, non_blocking=True)
                x_noisy_stft = x_noisy_stft.to(DEVICE, non_blocking=True)
            except RuntimeError as e:
                raise RuntimeError(f"Failed to move batch {batch_idx} to device {DEVICE}: {str(e)}")

            # Forward pass for main training
            fg1_wav = net(g1_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
          
            # Forward pass for regularization (no gradients needed)
            with torch.no_grad():
                fx_wav = net(x_noisy_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
                g1fx, g2fx = subsample2(fx_wav)
                g1fx, g2fx = g1fx.float(), g2fx.float()

            # Calculate loss (tensors already on correct device)
            loss = loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
            
            # Scale loss for gradient accumulation
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            loss.backward()
            accumulated_loss += loss.item()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Optional gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
                net.zero_grad()  # More efficient than optimizer.zero_grad()
                
                train_ep_loss += accumulated_loss
                accumulated_loss = 0.
                counter += 1
                
            # Periodic memory cleanup (every 50 batches)
            if batch_idx % 50 == 0 and batch_idx > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if counter > 0:
            train_ep_loss /= counter
        else:
            print("Warning: No valid batches processed")
            train_ep_loss = float('inf')
        
        return train_ep_loss
        
    except Exception as e:
        print(f"Error in training epoch: {str(e)}")
        # Emergency memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def test_epoch(net, test_loader, loss_fn, use_net=True):
    """Optimized test epoch with better error handling and memory management."""
    net.eval()
    test_ep_loss = 0.
    counter = 0
    
    try:
        with torch.no_grad():  # Critical: no gradients needed in test phase
            for batch_idx, (x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft) in enumerate(test_loader):
                # Validate batch data
                if any(tensor.numel() == 0 for tensor in [x_noisy_stft, g1_stft, g1_wav, g2_wav]):
                    print(f"Warning: Empty tensors in test batch {batch_idx}, skipping...")
                    continue
                    
                # Move tensors to device efficiently
                try:
                    g1_stft = g1_stft.to(DEVICE, non_blocking=True)
                    g1_wav = g1_wav.to(DEVICE, non_blocking=True)
                    g2_wav = g2_wav.to(DEVICE, non_blocking=True) 
                    x_noisy_stft = x_noisy_stft.to(DEVICE, non_blocking=True)
                except RuntimeError as e:
                    print(f"Warning: Failed to move test batch {batch_idx} to device: {str(e)}")
                    continue

                # Forward passes
                fg1_wav = net(g1_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
                fx_wav = net(x_noisy_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
                
                g1fx, g2fx = subsample2(fx_wav)
                g1fx, g2fx = g1fx.float(), g2fx.float()

                # Calculate loss (all tensors already on device)
                loss = loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
                test_ep_loss += loss.item()
                counter += 1

        if counter > 0:
            test_ep_loss /= counter
        else:
            print("Warning: No valid test batches processed")
            test_ep_loss = float('inf')

        print("Loss computation done...running metrics evaluation")

        # Run metrics evaluation
        testmet = getMetricsonLoader(test_loader, net, use_net)

        return test_ep_loss, testmet
        
    except Exception as e:
        print(f"Error in test epoch: {str(e)}")
        # Emergency memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def train(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):
    train_losses = []
    test_losses = []

    for e in tqdm(range(epochs)):

        train_loss = train_epoch(net, train_loader, loss_fn, optimizer)
        test_loss = 0
        scheduler.step()
        print("Saving model....")

        with torch.no_grad():
            test_loss, testmet = test_epoch(net, test_loader, loss_fn, use_net=True)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        with open(fixedpath + basepath + '/results.txt', "a") as f:
            f.write("Epoch :" + str(e + 1) + "\n" + str(testmet))
            f.write("\n")

        print("OPed to txt")

        torch.save(net.state_dict(), fixedpath + basepath + '/Weights/dc10_model_' + str(e + 1) + '.pth')
        torch.save(optimizer.state_dict(), fixedpath + basepath + '/Weights/dc10_opt_' + str(e + 1) + '.pth')

        print("Models saved")

        # Efficient memory management - only clear cache periodically
        if (e + 1) % 5 == 0:  # Clear cache every 5 epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("Epoch: {}/{}...".format(e+1, epochs),
                     "Loss: {:.6f}...".format(train_loss),
                     "Test Loss: {:.6f}".format(test_loss))
    return train_loss, test_loss

######################################## Train CONFI #########################################
test_dataset = SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)
train_dataset = SpeechDataset(train_input_files, train_target_files, N_FFT, HOP_LENGTH)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# For testing purpose
test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Efficient initial memory management
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Initialize model with error handling
try:
    dcunet = DCUnet10(N_FFT, HOP_LENGTH).to(DEVICE)
    print(f"Model initialized on {DEVICE}")
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    raise

# Optimized optimizer setup
optimizer = torch.optim.Adam(dcunet.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = RegularizedLoss(gamma=2 if noise_class == "white" else 1)  # Adaptive gamma
loss_fn = loss_fn.to(DEVICE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Model parameter count
total_params = sum(p.numel() for p in dcunet.parameters())
trainable_params = sum(p.numel() for p in dcunet.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# specify paths and uncomment to resume training from a given point
# model_checkpoint = torch.load(path_to_model)
# opt_checkpoint = torch.load(path_to_opt)
# dcunet.load_state_dict(model_checkpoint)
# optimizer.load_state_dict(opt_checkpoint)

print("Starting optimized training...")
train_losses, test_losses = train(dcunet, train_loader, test_loader, loss_fn, optimizer, scheduler, 20)
print(f"Training completed. Final train loss: {train_losses:.6f}, test loss: {test_losses:.6f}")

# Final cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Training session completed successfully!")

