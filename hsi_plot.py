import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def load_hyperspectral_data(mat_file):
    """
    Load hyperspectral data from a .mat file.
    
    Parameters:
        mat_file (str): Path to the .mat file containing the hyperspectral data.
    
    Returns:
        np.ndarray: The hyperspectral data array.
    """
    data = sio.loadmat(mat_file)
    # print(data)
    # Assuming the hyperspectral data is stored in the key 'data'
    hyperspectral_data = data.get('indian_pines', None)
    if hyperspectral_data is None:
        raise ValueError("The .mat file does not contain 'data' key.")
    return hyperspectral_data

def visualize_band_groups_as_rgb(hyperspectral_data, output_dir="output"):
    """
    Visualize and save selected groups of three bands of hyperspectral data as RGB images, arranged in a row with some spacing.
    
    Parameters:
        hyperspectral_data (np.ndarray): The hyperspectral data array (H, W, D).
        output_dir (str): Directory where the visualizations will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_bands = hyperspectral_data.shape[2]
    
    # Check if we have at least 21 bands
    if num_bands < 21:
        raise ValueError("The hyperspectral data does not contain 21 bands.")
    
    # Calculate how many rows of images we need
    num_images = min(num_bands // 3, 7)  # Limit to a max of 7 rows if there are more than 21 bands
    
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))  # Create a row of subplots
    
    # Iterate over the first 21 bands in groups of 3
    for i in range(0, 21, 3):
        # Ensure we are not accessing out-of-bounds bands
        if i + 2 >= num_bands:
            break
        
        # Select three consecutive bands for RGB visualization
        r_band = hyperspectral_data[:, :, i]   # Red channel (Band i)
        g_band = hyperspectral_data[:, :, i+1] # Green channel (Band i+1)
        b_band = hyperspectral_data[:, :, i+2] # Blue channel (Band i+2)

        # Apply logarithmic scaling to enhance contrast
        r_band_log = np.log1p(r_band)  # Apply log scaling to reduce the dynamic range
        g_band_log = np.log1p(g_band)  # Apply log scaling
        b_band_log = np.log1p(b_band)  # Apply log scaling

        # Normalize the log-scaled bands to [0, 255]
        r_band_normalized = np.uint8(255 * (r_band_log - np.min(r_band_log)) / (np.max(r_band_log) - np.min(r_band_log)))
        g_band_normalized = np.uint8(255 * (g_band_log - np.min(g_band_log)) / (np.max(g_band_log) - np.min(g_band_log)))
        b_band_normalized = np.uint8(255 * (b_band_log - np.min(b_band_log)) / (np.max(b_band_log) - np.min(b_band_log)))

        # Stack the bands into an RGB image
        rgb_image = np.stack((r_band_normalized, g_band_normalized, b_band_normalized), axis=-1)
        
        # Add the RGB image to the subplot
        ax = axes[i // 3]  # Select the appropriate subplot axis
        ax.imshow(rgb_image)
        ax.set_title(f"Bands {i+1}, {i+2}, {i+3} (RGB)")
        ax.axis('off')  # Hide axes

    # Adjust the layout to avoid overlap
    plt.subplots_adjust(wspace=0.3)  # Adjust horizontal space between subplots
    plt.tight_layout()  # Ensure proper spacing around the edges

    # Save the combined image
    plt.savefig(os.path.join(output_dir, "combined_rgb_bands.png"))
    plt.close()

    print(f"RGB images for the first 21 bands have been saved to {output_dir}.")

def plot_color_histogram(hyperspectral_data, band_indices, output_dir="output"):
    """
    Plot and save color histograms for selected bands (RGB channels).
    
    Parameters:
        hyperspectral_data (np.ndarray): The hyperspectral data array (H, W, D).
        band_indices (list of int): List of band indices to visualize.
        output_dir (str): Directory where the histograms will be saved.
    """
    # Assume RGB channels are made by combining specific bands
    if len(band_indices) < 3:
        raise ValueError("You need at least 3 bands for RGB color histogram.")
    
    rgb_data = hyperspectral_data[:, :, band_indices[:3]]  # Select the first 3 bands for RGB
    
    # Flatten the bands to create histograms for R, G, B channels
    r_channel = rgb_data[:, :, 0].flatten()
    g_channel = rgb_data[:, :, 1].flatten()
    b_channel = rgb_data[:, :, 2].flatten()

    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(r_channel, bins=50, color='red', alpha=0.6, label='Red Channel')
    plt.hist(g_channel, bins=50, color='green', alpha=0.6, label='Green Channel')
    plt.hist(b_channel, bins=50, color='blue', alpha=0.6, label='Blue Channel')
    
    plt.title("RGB Channel Color Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "color_histogram.png"))
    plt.close()

def plot_combined_band_histograms(hyperspectral_data, output_dir="output"):
    """
    Plot and save a combined histogram for the first 21 bands of hyperspectral data.
    
    Parameters:
        hyperspectral_data (np.ndarray): The hyperspectral data array (H, W, D).
        output_dir (str): Directory where the histograms will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the number of bands in the hyperspectral data
    num_bands = hyperspectral_data.shape[2]
    
    # Check if we have at least 21 bands
    if num_bands < 21:
        raise ValueError("The hyperspectral data does not contain 21 bands.")
    
    # Set up a figure for the combined histograms
    plt.figure(figsize=(14, 8))

    # Create a color palette for the histograms
    color_palette = sns.color_palette("Set2", n_colors=21)
    
    # Plot histograms for the first 21 bands
    for band_idx in range(21):
        band_data = hyperspectral_data[:, :, band_idx]  # Extract the current band
        
        # Flatten the band data for histogram plotting
        band_data_flattened = band_data.flatten()
        
        # Plot histogram for the current band with a specific color
        plt.hist(band_data_flattened, bins=50, alpha=0.7, color=color_palette[band_idx], label=f"Band {band_idx + 1}")
    
    # Adding titles and labels
    plt.title("Histograms of First 21 Bands of Hyperspectral Data", fontsize=16)
    plt.xlabel("Pixel Intensity", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    
    # Adding gridlines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adding legend with better placement to avoid overlap
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), title="Bands", fontsize=12)
    
    # Adjust the layout to prevent clipping of labels and legend
    plt.tight_layout()

    # Save the combined histogram plot
    plot_filename = os.path.join(output_dir, "combined_band_histograms.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"Combined histogram for the first 21 bands has been saved to {plot_filename}.")


def main():
    # File path to the .mat hyperspectral data
    mat_file = "/data1/userhome/luwen/Code/wzy/CAD2VSR/Indian_pines.mat"
    
    # Load hyperspectral data
    hyperspectral_data = load_hyperspectral_data(mat_file)
    
    # Select which bands to visualize (example: bands 10, 20, 30)
    band_indices = [10, 20, 30]  # Example: bands 10, 20, 30, you can change this
    
    # Visualize selected bands
    visualize_band_groups_as_rgb(hyperspectral_data)
    
    # Plot and save color distribution (assuming the bands correspond to RGB channels)
    plot_combined_band_histograms(hyperspectral_data, output_dir="output")
    
    print("Process completed! Visualizations and histograms saved.")

if __name__ == "__main__":
    main()
