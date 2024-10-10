from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
from datasets import (load_dataset, Features, Image, ClassLabel)
import datetime
import os
import argparse
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.optim as optim
from PIL import Image as im
import numpy as np

# Instead of the full (d, d, d) coskewness tensor, compute the coskewness for a slice of the tensor, specified by the start and end indices for the slice dimension. So for slice_start_index=0, slice_end_index=k, slice_dim=0, the output is a (k, d, d) tensor.
def compute_coskewness_slice(X, slice_start_index, slice_end_index, slice_dim=0):
    # X is a 2D tensor (n, d) where each row is a data point and each column is a dimension
    X_centered = X - X.mean(dim=0)
    # Slice the centered data along the specified dimension
    if slice_dim == 0:
        X_sliced = X_centered[:, slice_start_index:slice_end_index]
        coskewness_slice = torch.einsum('ij,ik,il->jkl', X_sliced, X_centered, X_centered) / X.shape[0]
    elif slice_dim == 1:
        X_sliced = X_centered[:, slice_start_index:slice_end_index]
        coskewness_slice = torch.einsum('ij,ik,il->jkl', X_centered, X_sliced, X_centered) / X.shape[0]
    elif slice_dim == 2:
        X_sliced = X_centered[:, slice_start_index:slice_end_index]
        coskewness_slice = torch.einsum('ij,ik,il->jkl', X_centered, X_centered, X_sliced) / X.shape[0]
    else:
        raise ValueError("slice_dim must be 0, 1, or 2")
    return coskewness_slice

def compute_covariance(X):
    # X is a 2D tensor where each row is a data point and each column is a dimension
    X_centered = X - X.mean(dim=0)
    covariance = X_centered.t() @ X_centered / X.shape[0]
    return covariance

def infer_columns(feats: Features) -> tuple[str, str]:
    # Infer the appropriate columns by their types 
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]

    assert len(img_cols) == 1, f"Found {len(img_cols)} image columns"
    assert len(label_cols) == 1, f"Found {len(label_cols)} label columns"

    return img_cols[0], label_cols[0]

def to_tensor(x):
    if isinstance(x, list):
        x = torch.tensor(x)
    if isinstance(x, im.Image):
        x = torch.tensor(np.array(x), dtype=torch.float32)
        if x.max() > 1:
            x /= 255.0
    return x

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', help='Device to use')
parser.add_argument('--n_steps', type=int, default=1500, help='Number of optimization steps')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use')
parser.add_argument('--n_fake_samples', type=int, default=1000, help='Number of fake samples to generate per class')
parser.add_argument('--save_dir', type=str, default="/mnt/ssd-1/cifar_learning_exps/gaussian_approximations/fake_data", help='Directory to save the fake datasets')
parser.add_argument('--split', type=str, default='train', help='Split to use')
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the optimization")
parser.add_argument("--slice_dim", type=int, default=0, help="Dimension to slice the coskewness tensor by")
parser.add_argument("--slice_size", type=int, default=1, help="Size of the slice of the coskewness tensor")
parser.add_argument("--test_slice", action="store_true", help="Test that the slice function works")
args = parser.parse_args()
device = args.device

if args.test_slice:
    # Create a random 3D tensor
    x = torch.randn(10000, 200).to(device)
    # Use the sliced coskewness function to compute the coskewness tensor in batches, and compare the results to the full coskewness tensor
    sliced_coskewness_tensors = torch.zeros((200, 200, 200)).to(device)
    for i in range(0, 200, 10):
        sliced_coskewness_tensors[i:i+10] += compute_coskewness_slice(x.to(torch.float64), i, i+10, 0)
    # Use compute_coskewness_slice with the full tensor selected as the slice
    full_coskewness_tensor = compute_coskewness_slice(x.to(torch.float64), 0, 200, 0)
    # Compare their values
    print(torch.norm(sliced_coskewness_tensors - full_coskewness_tensor))
    exit()

# Make save dir
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dataset_name = args.dataset.replace("/", "-")
save_dir = args.save_dir[:-1] if args.save_dir[-1] == "/" else args.save_dir
save_dir = save_dir + f"/{timestamp}_fake_{dataset_name}_{args.split}_data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

real_dataset = load_dataset(args.dataset, split=args.split)

img_col, label_col = infer_columns(real_dataset.features)
labels = real_dataset.features[label_col].names
print(f"Classes in '{args.dataset}': {labels}")
n_classes = len(labels)

# Infer the image size from the first image
h, w = real_dataset[0][img_col].size
print(f"Image size: {h} x {w}")

real_dataset_full_size = real_dataset.map(lambda x: {label_col: x[label_col], img_col: transforms.ToTensor()(x[img_col])})

# Initialize empty lists to hold tensors for each class
real_dataset_tensors_full_size = [[] for _ in range(n_classes)]

# Iterate through the dataset and append the images to the corresponding class list
for img_full_size, label in zip(real_dataset_full_size[img_col], real_dataset_full_size[label_col]):    
    img_full_size = to_tensor(img_full_size)
    real_dataset_tensors_full_size[label].append(img_full_size.flatten().unsqueeze(0))


for target_class in range(n_classes):
    print(f"Optimizing for class {target_class}")
    
    # Convert lists to tensors and move to device
    real_dataset_tensors_full_size[target_class] = torch.cat(real_dataset_tensors_full_size[target_class], dim=0).to(device)

    # Compute the means:
    real_dataset_tensor_mean_full_size = real_dataset_tensors_full_size[target_class].mean(dim=0)

    # Compute the covariances:
    real_dataset_tensor_full_size_cov = compute_covariance(real_dataset_tensors_full_size[target_class])

    print("Covariance tensor size:", real_dataset_tensor_full_size_cov.size())

    # Cannot compute a static coskewness tensor for the full size dataset because it's too large (d^3)
    # We will compute the coskewness for a slice of the tensor, specified by the start and end indices for the slice dimension.

    # Initialize a fake dataset by sampling from a gaussian with real_dataset_tensor_mean_full_size and real_dataset_tensor_full_size_cov covariance
    # Need to add a small value to the diagonals of real_dataset_tensor_full_size_cov to make it positive definite
    real_dataset_tensor_full_size_cov = real_dataset_tensor_full_size_cov + 2e-6 * torch.eye(real_dataset_tensor_full_size_cov.size(0)).to(device)
    while True:
        try:
            mvn = MultivariateNormal(real_dataset_tensor_mean_full_size, real_dataset_tensor_full_size_cov)
            break
        except:
            real_dataset_tensor_full_size_cov += 2e-6 * torch.eye(real_dataset_tensor_full_size_cov.size(0)).to(device)

    fake_dataset_tensor = mvn.sample((args.n_fake_samples,))
    fake_dataset_tensor = torch.clamp(fake_dataset_tensor, 0, 1)
    del real_dataset_tensor_full_size_cov

    # Optimize the values of fake_dataset_tensor so its statistics are close to the real dataset
    fake_dataset_tensor.requires_grad = True
    optimizer = optim.Adam([fake_dataset_tensor], lr=args.lr)
    for i in tqdm(range(args.n_steps)):
        optimizer.zero_grad()
        # Compute the mean of fake_dataset_tensor
        fake_dataset_tensor_mean = fake_dataset_tensor.mean(dim=0)

        # Compute the convariance matrix of fake_dataset_tensor
        fake_dataset_tensor_cov = compute_covariance(fake_dataset_tensor)

        # Iterate through the slices of the coskewness tensor
        num_slices = (h + args.slice_size - 1) // args.slice_size
        for j in range(0,h,args.slice_size):
            # Compute the start and end indices for the slice dimension
            start_index = (j * args.slice_size) % h
            end_index = min(start_index + args.slice_size, h)

            # Compute a slice of the coskewness tensor of fake_dataset_tensor
            fake_dataset_tensor_coskew = compute_coskewness_slice(fake_dataset_tensor, start_index, end_index, args.slice_dim)

            # Compute the matching slice of the coskewness tensor of the real dataset
            real_dataset_tensor_full_size_coskew = compute_coskewness_slice(real_dataset_tensors_full_size[target_class], start_index, end_index, args.slice_dim)

            # Compute the loss for the slice
            coskew_loss = (fake_dataset_tensor_coskew - real_dataset_tensor_full_size_coskew).norm() / real_dataset_tensor_full_size_coskew.numel()
            coskew_loss = coskew_loss / num_slices

            # Backpropagate the loss and accumulate the gradients
            coskew_loss.backward()

        # Compute the loss
        means_loss = (fake_dataset_tensor_mean - real_dataset_tensor_mean_full_size).norm() / real_dataset_tensor_mean_full_size.numel()

        cov_loss = (fake_dataset_tensor_cov - real_dataset_tensor_full_size_cov).norm() / real_dataset_tensor_full_size_cov.numel()
                
        bounds_loss = (torch.relu(fake_dataset_tensor - 1) + torch.relu(-fake_dataset_tensor)).mean()

        loss = means_loss + cov_loss + bounds_loss
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Total Loss: {}".format(loss.item()))
            print("Sublosses: means_loss: {}, cov_loss: {}, coskew_loss: {}, bounds_loss: {}".format(means_loss.item(), cov_loss.item(), coskew_loss.item(), bounds_loss.item()))

    # Save the fake dataset
    torch.save(fake_dataset_tensor, save_dir + f"/downscaling_fake_{dataset_name}_{args.split}_class_{target_class}.pt")

