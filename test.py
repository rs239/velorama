import numpy as np

import matplotlib.pyplot as plt

# Create a new figure
plt.figure(figsize=(8, 6))

# Draw some simple lines
x = np.linspace(0, 10, 100)
plt.plot(x, x, label='Linear')  # straight line
plt.plot(x, np.sin(x), label='Sine')  # sine wave
plt.plot(x, x**2/10, label='Quadratic')  # parabola

# Add labels and title
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Simple Lines Plot')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()