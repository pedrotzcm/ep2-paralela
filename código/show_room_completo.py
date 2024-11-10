import numpy as np
import matplotlib.pyplot as plt

# Load heat data from file
heat_data = np.loadtxt("room.txt")
heat_data_cuda = np.loadtxt("codigo_cuda/roomcuda.txt")

extent = [0, 10, 0, 10]

# Create the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot heatmap for room.txt
cax1 = axes[0].imshow(
    heat_data, cmap="hot", interpolation="nearest", extent=extent, origin="lower"
)
axes[0].set_title("Heat Distribution (CPU)")
axes[0].set_xlabel("X Coordinate")
axes[0].set_ylabel("Y Coordinate")
fig.colorbar(cax1, ax=axes[0], label="Temperature")  # Add colorbar for the first plot

# Plot heatmap for roomcuda.txt
cax2 = axes[1].imshow(
    heat_data_cuda, cmap="hot", interpolation="nearest", extent=extent, origin="lower"
)
axes[1].set_title("Heat Distribution (GPU)")
axes[1].set_xlabel("X Coordinate")
axes[1].set_ylabel("Y Coordinate")
fig.colorbar(cax2, ax=axes[1], label="Temperature")  # Add colorbar for the second plot

# Save the plot as an image file
plt.tight_layout()  # Adjust layout to avoid overlap
plt.savefig("heat_distribution_comparison.png")  # Saves as a PNG file in the current directory

# Show the plot
plt.show()
