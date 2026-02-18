import h5py
f = h5py.File("my_dataset.h5", "r")

# Get metadata for first episode of 'reach'
ep = f["reach/episode_0000"]
print(ep.attrs["scenario_disruption_type"])

# Plot separation distance
import matplotlib.pyplot as plt
plt.plot(ep["safety/min_separation"][:])
plt.show()