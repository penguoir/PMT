import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import random

a = [200]

for i in np.arange (0, 299, 1):
    if i%20==0:
        a.append(random.random()*5+255)

    elif random.randint(0, 289) % 257 == 0:
        a.append(random.random() * 3 + 228)
        a.append(random.random() * 3 + 228)
        a.append(random.random() * 3 + 228)
        a.append(random.random() * 3 + 228)
        a.append(random.random() * 3 + 228)
        a.append(random.random() * 3 + 228)
        print(i)

    else:
        a.append(random.random()*10+295)


np.savez('oof', range(len(a)), a)
plt.plot(np.arange(0, len(a), 1), a, '-')
plt.show()