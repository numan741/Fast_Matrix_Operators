from Vect import Vect
from time import perf_counter

import numpy as np
i=100
A=np.zeros(shape=(i,))
B=np.zeros(shape=(i,))
for i in range(i):
   A[i] = i
   B[i] = 2*i


V1=Vect(A)
V2=Vect(B)


start = perf_counter()
V3=V1+(V2*222)+V1-(V2/2)+V1+V1
duration = perf_counter() - start

print(V3.toNumpy())
print('Module took {:.4f} Milliseconds\n\n'.format(duration*1000))

start = perf_counter()
a=A+(B*222)+A-(B/2)+A+A
duration = perf_counter() - start


print(a)
print('Numpy took {:.4f} Milliseconds\n\n'.format(duration*1000))



