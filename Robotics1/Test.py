import numpy as np
import matplotlib.pyplot as plt

# Load the data from 'Dangerous.csv'
data = np.genfromtxt('Dangerous.csv', delimiter=',', names=True, dtype=None, encoding=None)

# Access the 'time' and 'yd' columns as NumPy arrays.
xd = data['time']
yd = data['yd']

# Calculate the maximum time and create a linspace for x values
max_time = xd.max()
xq = np.linspace(0, max_time, len(xd))

# Gaussian processing
# You can replace these parameters with your specific Gaussian function
mean = np.mean(xd)
std_dev = np.std(xd)
gaussian = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xq - mean) / std_dev)**2)

# Plot the data and Gaussian curve
plt.figure(figsize=(8, 4))
plt.plot(xd, yd, label='Data', marker='o')
plt.plot(xq, gaussian, label='Gaussian', linestyle='--')

# Add uncertainty fill
plt.fill_between(xq, yd - 2 * std_dev, yd + 2 * std_dev, color='gray', alpha=0.5, label='Uncertainty')

plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('Data with Gaussian Curve and Uncertainty')
plt.grid()
plt.show()
