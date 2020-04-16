import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return np.piecewise(x, [x < 2, x >= 2], [lambda x: a*x, lambda x: -b*x+c])
# Define the data to be fit with some noise:

real_func_base = lambda x: func(x, 3/2, 3/2, 6)

xdata = np.linspace(0, 4, 50)
y = real_func_base(xdata)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')
# Fit for the parameters a, b, c of the function func:

popt, pcov = curve_fit(func, xdata, ydata)
print(popt)
print(pcov)

plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
# Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:

print(np.vectorize(func)(xdata, *popt))
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))

print(popt)
print(pcov)


plt.plot(xdata, np.vectorize(func)(xdata, *popt), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
