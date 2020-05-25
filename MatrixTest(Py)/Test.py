from Mat import Mat
from Vect import Vect
from time import perf_counter

import numpy as np
i=3
A=np.array([[1,34,45],[1,4,78],[3,5,7]])
print(A)
M1=Mat(A)
M2=Mat(A)


start = perf_counter()
V3=M1.inv()
duration = perf_counter() - start
print(V3.toNumpy())
print('Module took {:.4f} Milliseconds\n\n'.format(duration*1000))

start = perf_counter()
a=np.linalg.inv(A)
duration = perf_counter() - start
print(a)
print('Numpy took {:.4f} Milliseconds\n\n'.format(duration*1000))


