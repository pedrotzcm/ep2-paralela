import numpy as np
import matplotlib.pyplot as plt

# Load heat data from file
heat_data = np.loadtxt("room.txt")

extent = [0, 10, 0, 10]

# Plot heatmap
plt.imshow(
    heat_data, cmap="hot", interpolation="nearest", extent=extent, origin="lower"
)
plt.colorbar(label="Temperature")
plt.title("Heat Distribution")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

# Save the plot as an image file
plt.savefig("heat_distribution.png")  # Saves as a PNG file in the current directory
