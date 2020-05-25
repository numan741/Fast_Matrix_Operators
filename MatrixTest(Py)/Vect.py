import cVect 
import numpy as np
class Vect():
    def __init__(self,num,op=0):
        if op==0:
            self.vectCapsule = cVect.construct(num)
        else:
            self.vectCapsule=num
    def __add__(self, Vect2):
       return  Vect(cVect.AddOperator(self.vectCapsule, Vect2.vectCapsule),1)
        
    def __sub__(self, Vect2):
       return Vect(cVect.SubOperator(self.vectCapsule, Vect2.vectCapsule),1)

    def __mul__(self,double):
        return Vect(cVect.ScalarMul(self.vectCapsule,double),1)
    def __truediv__(self,double):
        return Vect(cVect.ScalarDiv(self.vectCapsule,double),1)
    def toNumpy(self): 
        return cVect.toNum(self.vectCapsule)

    def __delete__(self):
        cVect.delete_object(self.vectCapsule)


