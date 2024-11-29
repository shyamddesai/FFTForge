from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import csr_matrix, save_npz
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time

# Load an image and convert it to grayscale
def load_image(image_path):
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert("L")
    return np.array(image)

# Pad the image to the next power of 2
def pad_to_power_of_two(image):
    rows, cols = image.shape
    new_rows = 2**int(np.ceil(np.log2(rows)))
    new_cols = 2**int(np.ceil(np.log2(cols)))
    padded_image = np.zeros((new_rows, new_cols), dtype=image.dtype)
    padded_image[:rows, :cols] = image
    print(f"Padded image from ({rows}, {cols}) to ({new_rows}, {new_cols})")
    return padded_image

# Crop the image back to its original dimensions
def crop_to_original(image, original_shape):
    rows, cols = original_shape
    return image[:rows, :cols]

# Naive 1D-DFT
def dft_1d(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return result

# Naive 2D-DFT
def dft_2d(image):
    print("Performing 2D-DFT (Naive)...")
    rows, cols = image.shape
    dft_rows = np.array([dft_1d(row) for row in image])  # Row-wise DFT
    dft_result = np.array([dft_1d(dft_rows[:, col]) for col in range(cols)]).T  # Column-wise DFT
    return dft_result

# Cooley-Tukey 1D-FFT Implementation
def fft_1d(signal):
    N = len(signal)
    if N <= 1:  # Base case for recursion
        return signal
    if N % 2 != 0:
        raise ValueError("Input size must be a power of 2")
    
    even = fft_1d(signal[::2])  # FFT of even-indexed elements
    odd = fft_1d(signal[1::2])  # FFT of odd-indexed elements
    T = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    return np.concatenate([even + T, even - T])

# Inverse FFT (1D) using the Cooley-Tukey Method
def ifft_1d(signal):
    N = len(signal)
    signal_conj = np.conjugate(signal)
    result = fft_1d(signal_conj)
    return np.conjugate(result) / N

# 2D-FFT Implementation
def fft_2d(image):
    print("Performing 2D-FFT...")
    rows_transformed = np.array([fft_1d(row) for row in image])
    cols_transformed = np.array([fft_1d(rows_transformed[:, col]) for col in range(image.shape[1])]).T
    return cols_transformed

# 2D-Inverse FFT Implementation
def ifft_2d(image):
    print("Performing 2D-Inverse FFT...")
    rows_transformed = np.array([ifft_1d(row) for row in image])
    cols_transformed = np.array([ifft_1d(rows_transformed[:, col]) for col in range(image.shape[1])]).T
    return cols_transformed

# Perform FFT and return magnitude and phase
def perform_fft(image):
    padded_image = pad_to_power_of_two(image)
    fft_result = fft_2d(padded_image)
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    return magnitude, phase

# Perform Inverse FFT using magnitude and phase
def perform_inverse_fft(magnitude, phase, original_shape):
    complex_spectrum = magnitude * np.exp(1j * phase)
    reconstructed_padded = np.abs(ifft_2d(complex_spectrum))
    return crop_to_original(reconstructed_padded, original_shape)

# Apply a low-pass filter
def apply_low_pass_filter(fft_im, keep_ratio):
    rows, cols = fft_im.shape
    crow, ccol = rows // 2, cols // 2

    # Calculate the keep region
    r_keep = int(keep_ratio * rows / 2)
    c_keep = int(keep_ratio * cols / 2)

    # Create a mask for the low-pass filter
    mask = np.zeros_like(fft_im, dtype=bool)
    mask[crow - r_keep:crow + r_keep, ccol - c_keep:ccol + c_keep] = True

    # Apply the mask to the FFT (preserving the phase)
    filtered_fft = fft_im * mask
    return filtered_fft

# Compress the magnitude by zeroing out smaller coefficients
def compress_magnitude(magnitude, compression_level):
    print(f"\nCompressing with level: {compression_level * 100}%")
    threshold = np.percentile(magnitude, 100 * compression_level)  # Determine threshold
    compressed_magnitude = magnitude.copy()
    compressed_magnitude[compressed_magnitude < threshold] = 0  # Zero out smaller coefficients
    return compressed_magnitude

# Save and display results for our FFT method
def display_fft_magnitude(image, mode):
    results_dir = f"results/mode_{mode}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving results in directory: {results_dir}")

    # Compute our implementation of FFT magnitude
    magnitude, _ = perform_fft(image)
    cropped_magnitude = crop_to_original(magnitude, image.shape) # Crop to original size for better visualization against NumPy
    log_scaled_magnitude = np.log1p(cropped_magnitude)

    # Save the images
    plt.imsave(os.path.join(results_dir, "original_image.png"), image, cmap="gray")
    plt.imsave(os.path.join(results_dir, "log_scaled_custom_fft_magnitude.png"), log_scaled_magnitude, cmap="viridis")

    # Plot the original and transformed images
    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Original")
    ax1.imshow(image, cmap="gray")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Log-scaled Custom FFT Magnitude")
    im = ax2.imshow(log_scaled_magnitude, cmap="viridis", norm=LogNorm(vmin=1))

    # Create a colorbar with more ticks and matching width
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding for width
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Intensity", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    plt.suptitle("Custom FFT Transform Results", fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fft_transform_results_custom_fft.png"))
    plt.show()

# Save and display results for NumPy FFT method
def display_numpy_fft_magnitude(image, mode):
    results_dir = f"results/mode_{mode}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving results in directory: {results_dir}")

    # Compute NumPy FFT magnitude
    numpy_magnitude = np.abs(np.fft.fft2(image))
    log_scaled_magnitude = np.log1p(numpy_magnitude)

    # Save the image
    plt.imsave(os.path.join(results_dir, "log_scaled_numpy_fft_magnitude.png"), log_scaled_magnitude, cmap="viridis")

    # Plot the original and transformed images
    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Original")
    ax1.imshow(image, cmap="gray")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Log-scaled NumPy FFT Magnitude")
    im = ax2.imshow(log_scaled_magnitude, cmap="viridis", norm=LogNorm(vmin=1))
 
    # Create a colorbar with more ticks and matching width
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding for width
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Intensity", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.suptitle("NumPy FFT Transform Results", fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fft_transform_results_numpy_fft.png"))
    plt.show()

# Perform denoising and display results
def display_denoised_fft(image, keep_fraction, mode):
    results_dir = f"results/mode_{mode}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving results in directory: {results_dir}")

    # Perform FFT
    padded_image = pad_to_power_of_two(image)
    # fft_transformed = np.fft.fft2(padded_image)
    fft_transformed = fft_2d(padded_image)

    # Apply thresholding to the FFT-transformed image
    r, c = fft_transformed.shape
    fft_filtered = fft_transformed.copy()
    fft_filtered[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    fft_filtered[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0

    # Perform inverse FFT
    # denoised_padded = np.abs(np.fft.ifft2(fft_filtered))
    denoised_padded = np.abs(ifft_2d(fft_filtered)) 
    denoised_image = crop_to_original(denoised_padded, image.shape)

    # Normalize and scale the image for visualization
    denoised_image = (denoised_image - denoised_image.min()) / (denoised_image.max() - denoised_image.min())
    denoised_image *= 255
    denoised_image = denoised_image.astype(np.uint8)

    # Save images
    plt.imsave(os.path.join(results_dir, "original_image.png"), image, cmap="gray")
    plt.imsave(os.path.join(results_dir, "denoised_image_{:.2f}.png".format(keep_fraction)), denoised_image, cmap="gray")

    # Plot the original and denoised images
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Original")
    ax1.imshow(image, cmap="gray")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title(f"Denoised (Keep Ratio: {keep_fraction * 100:.1f}%)")
    ax2.imshow(denoised_image, cmap="gray")
    ax2.axis("off")

    plt.suptitle(f"Denoising with Keep Ratio: {keep_fraction * 100:.1f}%", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(os.path.join(results_dir, "denoising_results_{:.2f}.png".format(keep_fraction)))
    plt.show()

# Save the compressed images and their sparse matrices for various compression levels
def save_and_display_compression_results(image, magnitude, phase, original_shape, results_dir):
    compression_levels = [0.45, 0.8, 0.9, 0.97, 0.999]  # Compression levels to apply
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))  # Fixed figure size for clarity

    # Save and display original image (0% compression)
    sparse_matrix = csr_matrix(magnitude)
    original_file_path = os.path.join(results_dir, "sparse_matrix_original.npz")
    save_npz(original_file_path, sparse_matrix)
    original_size = os.path.getsize(original_file_path)
    original_size_str = f"{original_size:,}"  # Add commas to size
    original_nonzeros = sparse_matrix.nnz  # Number of non-zero elements

    # Save original image
    original_image_path = os.path.join(results_dir, "original_image.png")
    plt.imsave(original_image_path, image, cmap="gray")
    print(f"Original image saved: {original_image_path}")
    print(f"Original matrix size: {original_size_str} bytes, Non-zero elements: {original_nonzeros}")

    # Plot original image
    plt.subplot(2, 3, 1)  # 2 rows, 3 columns for 6 images (original + 5 compression levels)
    plt.imshow(image, cmap="gray")
    plt.title(f"Original\nMatrix Size: {original_size_str} bytes")
    plt.xticks([])
    plt.yticks([])

    # Iterate over compression levels
    for idx, level in enumerate(compression_levels, start=2):
        # Compress the magnitude
        compressed_magnitude = compress_magnitude(magnitude, level)

        # Save the sparse matrix
        sparse_matrix = csr_matrix(compressed_magnitude)
        sparse_file_path = os.path.join(results_dir, f"sparse_matrix_{int(level * 100)}.npz")
        save_npz(sparse_file_path, sparse_matrix)
        compressed_size = os.path.getsize(sparse_file_path)
        compressed_size_str = f"{compressed_size:,}"
        compressed_nonzeros = sparse_matrix.nnz  # Number of non-zero elements

        # Reconstruct the image using the compressed magnitude
        reconstructed_image = perform_inverse_fft(compressed_magnitude, phase, original_shape)

        # Save the reconstructed image
        compressed_image_path = os.path.join(results_dir, f"compressed_image_{int(level * 100)}.png")
        plt.imsave(compressed_image_path, reconstructed_image, cmap="gray")
        print(f"Compressed image saved: {compressed_image_path}")
        print(f"Compressed {level * 100:.1f}%: Matrix Size = {compressed_size_str} bytes, Non-zero elements: {compressed_nonzeros}")

        # Plot the reconstructed image
        plt.subplot(2, 3, idx)
        plt.imshow(reconstructed_image, cmap="gray")
        plt.title(f"Compressed {level * 100}%\nMatrix Size: {compressed_size_str} bytes")
        plt.xticks([])
        plt.yticks([])

    # Adjust layout and save the final plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ensure proper spacing
    plot_path = os.path.join(results_dir, "compression_results.png")
    plt.savefig(plot_path)
    print(f"Compression results plot saved: {plot_path}")
    plt.show()
    
# Compare runtime of Naive DFT and FFT
def plot_runtime_graphs():
    sizes = [2**5, 2**6, 2**7, 2**8]  # Image sizes 32, 64, 128, 256 to test
    dft_means = []
    fft_means = []
    dft_variances = []
    fft_variances = []
    dft_stddevs = []
    fft_stddevs = []

    # Perform multiple runs for statistical analysis
    num_trials = 10
    print("Performing runtime analysis with statistical measures...")

    for size in sizes:
        print(f"\n======== Processing size: {size}x{size} ========")
        image = np.random.rand(size, size)

        # Run Naive DFT multiple times
        dft_run_times = []
        for trial in range(1, num_trials + 1):
            start_time = time.time()
            dft_2d(image)
            elapsed_time = time.time() - start_time
            dft_run_times.append(elapsed_time)
            print(f"  Naive DFT Trial {trial}/{num_trials}: {elapsed_time:.6f} s")
        dft_means.append(np.mean(dft_run_times))
        dft_variances.append(np.var(dft_run_times))
        dft_stddevs.append(np.std(dft_run_times))

        # Run FFT multiple times
        fft_run_times = []
        for trial in range(1, num_trials + 1):
            start_time = time.time()
            fft_2d(image)
            elapsed_time = time.time() - start_time
            fft_run_times.append(elapsed_time)
            print(f"  FFT Trial {trial}/{num_trials}: {elapsed_time:.6f} s")
        fft_means.append(np.mean(fft_run_times))
        fft_variances.append(np.var(fft_run_times))
        fft_stddevs.append(np.std(fft_run_times))

    # Print means and variances
    print("\nRuntime Statistics")
    for idx, size in enumerate(sizes):
        print(f"Size: {size}x{size}")
        print(f"  Naive DFT -> Mean: {dft_means[idx]:.6f} s, Variance: {dft_variances[idx]:.6e}, Std Dev: {dft_stddevs[idx]:.6e}")
        print(f"  FFT       -> Mean: {fft_means[idx]:.6f} s, Variance: {fft_variances[idx]:.6e}, Std Dev: {fft_stddevs[idx]:.6e}")

    # Plot mean runtimes with 97% confidence interval (~2.17 standard deviations)
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        sizes, dft_means, yerr=[2.17 * std for std in dft_stddevs], label="2D-DFT (Naive)", fmt='o-', color='red', capsize=5
    )
    plt.errorbar(
        sizes, fft_means, yerr=[2.17 * std for std in fft_stddevs], label="2D-FFT (Cooley-Tukey)", fmt='o-', color='blue', capsize=5
    )

    plt.xticks(sizes) # Set x-ticks to image sizes
    plt.xlabel("Image Size (NxN)")
    plt.ylabel(f"Mean Runtime (s) from {num_trials} Trials")
    plt.title("Runtime Comparison: Naive DFT vs FFT")
    plt.legend()
    plt.grid()

    # Save the plot
    os.makedirs("results/mode_4", exist_ok=True)
    plt.savefig("results/mode_4/runtime_comparison.png")
    print("\nRuntime comparison plot saved to results/mode_4/runtime_comparison.png")
    plt.show()

# ====================================================================================================

def main():
    parser = argparse.ArgumentParser(description="Perform FFT-based tasks on an image.")
    parser.add_argument("-m", "--mode", type=int, default=1, help="Mode of operation: 1 (Fast), 2 (Denoise), 3 (Compress), 4 (Runtime)")
    parser.add_argument("-i", "--image", type=str, default="moonlanding.png", help="Path to the input image file.")
    args = parser.parse_args()

    image = load_image(args.image)
    original_shape = image.shape

    if args.mode == 1:
        display_fft_magnitude(image, mode=1) # Our FFT implementation
        display_numpy_fft_magnitude(image, mode=1) # Compare with NumPy FFT
        return

    elif args.mode == 2:
        threshold_factors = [0.01, 0.05, 0.075, 0.1, 0.2, 0.5, 0.9]
        for keep_fraction in threshold_factors:
            display_denoised_fft(image, keep_fraction, mode=2)
        return

    elif args.mode == 3:
        magnitude, phase = perform_fft(image)
        save_and_display_compression_results(image, magnitude, phase, original_shape, "results/mode_3")
        return
    
    elif args.mode == 4:
        plot_runtime_graphs()
        return

if __name__ == "__main__":
    main()

"""
Usage Instructions
-------------------
1. Fast Mode (Default):
   - Computes the FFT of the given image and reconstructs it without any modification.
   - Saves and displays the original image, FFT magnitude, and reconstructed image.
   Command:
   python fft.py -m 1 -i moonlanding.png

2. Denoise Mode:
   - Applies a low-pass filter to remove high-frequency noise from the FFT.
   - Reconstructs and saves the denoised image.
   Command:
   python fft.py -m 2 -i moonlanding.png

3. Compress Mode:
   - Compresses the image by retaining the highest magnitude components.
   - Saves and displays the reconstructed compressed image.
   Command:
   python fft.py -m 3 -i moonlanding.png

4. Plot Runtime Graphs:
   - Evaluates the runtime of 2D-DFT for images of varying sizes.
   - Plots and saves the runtime graph in the results directory.
   Command:
   python fft.py -m 4
"""