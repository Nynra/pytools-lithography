import matplotlib.pyplot as plt
import numpy as np

def A(x):
    return 3 * np.cos(2 * np.pi * x)

def B(x):
    return 2 * np.sin(2 * np.pi * x)

def C(x):
    return A(x) + B(x)


x = np.linspace(0, 1, 100)
y1 = A(x)
y2 = B(x)
y3 = C(x)

plt.plot(x, y1, label="A(x)")
plt.plot(x, y2, label="B(x)")
plt.plot(x, y3, label="C(x)")
plt.legend()
plt.grid(True)
plt.show()
