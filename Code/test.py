import numpy as np
from scipy import integrate

import matplotlib.pyplot as plt
import numpy as np

def f1(x):
   y = np.sin(2 * np.pi * f * x / Fs)
   return y

Fs = 8000
f = 5
sample = 8000
x = np.arange(sample)
y1 = f1(x)
 
result = integrate.romberg(f1,0, 8000, show=True,divmax=10000)
print result

plt.plot(x, y1)
plt.xlabel('voltage(V)')
plt.ylabel('sample(n)')
plt.show()
