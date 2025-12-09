import matplotlib.pyplot as plt
import numpy as np

# Generate some test data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a simple plot
plt.plot(x, y, label='sin(x)')
plt.title("Matplotlib Test")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()

# Show the plot
plt.show()
