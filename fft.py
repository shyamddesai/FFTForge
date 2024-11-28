import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
import argparse
import os
import time
from scipy.sparse import csr_matrix, save_npz

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
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# Inverse FFT (1D) using the Cooley-Tukey Method
def ifft_1d(signal):
    N = len(signal)
    signal_conj = np.conjugate(signal)
    result = fft_1d(signal_conj)
    return np.conjugate(result) / N

# 2D-FFT Implementation
def fft_2d(image):
    print("Performing 2D-FFT...")
    rows, cols = image.shape
    fft_rows = np.array([fft_1d(row) for row in image])  # Row-wise FFT
    fft_result = np.array([fft_1d(fft_rows[:, col]) for col in range(cols)]).T  # Column-wise FFT
    return fft_result

# 2D-Inverse FFT Implementation
def ifft_2d(image):
    print("Performing 2D-Inverse FFT...")
    rows, cols = image.shape
    ifft_rows = np.array([ifft_1d(row) for row in image])  # Row-wise Inverse FFT
    ifft_result = np.array([ifft_1d(ifft_rows[:, col]) for col in range(cols)]).T  # Column-wise Inverse FFT
    return ifft_result

# Apply a low-pass filter
def apply_low_pass_filter(magnitude, cutoff_frequency):
    print(f"Applying low-pass filter with cutoff frequency: {cutoff_frequency}")
    rows, cols = magnitude.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros_like(magnitude)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= cutoff_frequency:
                mask[i, j] = 1
    return magnitude * mask

# Perform FFT and return magnitude and phase
def perform_fft(image):
    padded_image = pad_to_power_of_two(image)
    fft_result = fft_2d(padded_image)
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    return magnitude, phase, fft_result, padded_image.shape

# Perform Inverse FFT using magnitude and phase
def perform_inverse_fft(magnitude, phase, original_shape):
    complex_spectrum = magnitude * np.exp(1j * phase)
    reconstructed_padded = np.abs(ifft_2d(complex_spectrum))
    return crop_to_original(reconstructed_padded, original_shape)

# Save and display results for different modes
def save_and_display_results(image, result_list, mode):
    results_dir = f"results/mode_{mode}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Saving results in directory: {results_dir}")

    for idx, (title, data, cmap) in enumerate(result_list):
        file_path = os.path.join(results_dir, f"{title}.png")
        plt.imsave(file_path, data, cmap=cmap)
        print(f"Saved: {file_path}")

    # Display the results
    plt.figure(figsize=(12, 8))
    for idx, (title, data, cmap) in enumerate(result_list):
        plt.subplot(1, len(result_list), idx + 1)
        plt.title(title)
        plt.imshow(data, cmap=cmap, norm=LogNorm() if "Magnitude" in title else None)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Compress the FFT magnitude by retaining a specified fraction of coefficients
def compress_magnitude(magnitude, compression_level):
    print(f"\nCompressing with level: {compression_level * 100}%")
    threshold = np.percentile(magnitude, 100 * compression_level)  # Determine threshold
    compressed_magnitude = magnitude.copy()
    compressed_magnitude[compressed_magnitude < threshold] = 0  # Zero out smaller coefficients
    return compressed_magnitude

# Save the compressed images and their sparse matrices for various compression levels
def save_and_display_compression_results(image, magnitude, phase, original_shape, results_dir):
    compression_levels = [0.45, 0.96, 0.97, 0.99, 0.999]  # Compression levels to apply
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
    sizes = [32, 64, 128, 256]
    dft_times = []
    fft_times = []

    for size in sizes:
        print(f"Processing size: {size}x{size}")
        image = np.random.rand(size, size)

        # Naive DFT Runtime
        start_time = time.time()
        dft_2d(image)
        dft_times.append(time.time() - start_time)

        # FFT Runtime
        start_time = time.time()
        fft_2d(image)
        fft_times.append(time.time() - start_time)

    plt.plot(sizes, dft_times, label="2D-DFT (Naive)", marker='o')
    plt.plot(sizes, fft_times, label="2D-FFT (Cooley-Tukey)", marker='o')
    plt.xlabel("Image Size")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Comparison: Naive DFT vs FFT")
    plt.legend()
    plt.grid()
    os.makedirs("results/mode_4", exist_ok=True)
    plt.savefig("results/mode_4/runtime_comparison.png")
    plt.show()

# ====================================================================================================

def main():
    parser = argparse.ArgumentParser(description="Perform FFT-based tasks on an image.")
    parser.add_argument("-m", "--mode", type=int, default=1, help="Mode of operation: 1 (Fast), 2 (Denoise), 3 (Compress), 4 (Runtime)")
    parser.add_argument("-i", "--image", type=str, default="moonlanding.png", help="Path to the input image file.")
    args = parser.parse_args()

    if args.mode == 4:
        plot_runtime_graphs()
        return

    image = load_image(args.image)
    original_shape = image.shape
    magnitude, phase, fft_result, padded_shape = perform_fft(image)

    if args.mode == 1:
        reconstructed_image = perform_inverse_fft(magnitude, phase, original_shape)
        result_list = [
            ("Original", image, "gray"),
            ("Log-scaled FFT Magnitude", np.log(1 + magnitude), "gray")
        ]
        save_and_display_results(image, result_list, mode=1)

    elif args.mode == 2:
        cutoff_frequency = 50
        filtered_magnitude = apply_low_pass_filter(magnitude, cutoff_frequency)
        reconstructed_image = perform_inverse_fft(filtered_magnitude, phase, original_shape)
        print(f"Non-zeros after filtering: {np.count_nonzero(filtered_magnitude)}")
        result_list = [
            ("Original", image, "gray"),
            ("Denoised", reconstructed_image, "gray")
        ]
        save_and_display_results(image, result_list, mode=2)

    elif args.mode == 3:
        save_and_display_compression_results(image, magnitude, phase, original_shape, "results/mode_3")

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