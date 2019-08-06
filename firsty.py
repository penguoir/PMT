import numpy as nm
import matplotlib.pyplot as plt
import scipy.optimize as norm


def func(xn, a, b, c):
    return a*nm.exp(-0.5*(xn-b)**2/c**2)


data = nm.load('oof.npz')

for i in data:
    print(i)
x = data['arr_0']
y = data['arr_1']
popt, pcov = norm.curve_fit(f= func, xdata= x, ydata = y)
print (popt)
plt.plot(x , func(x, popt[0],popt[1],popt[2]), '-')
plt.plot(x, y, '-')
plt.show()
