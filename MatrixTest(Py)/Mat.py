import cMat 
from Vect import Vect
import numpy as np
class Mat():
    def __init__(self,num,op=0):
        if op==0:
            self.MatCapsule = cMat.construct(num)
        else:
            self.MatCapsule=num
    def __add__(self, Mat2):
       return  Mat(cMat.AddOperator(self.MatCapsule, Mat2.MatCapsule),1)
        
    def __sub__(self, Mat2):
       return Mat(cMat.SubOperator(self.MatCapsule, Mat2.MatCapsule),1)

    def __mul__(self,double):
        if(isinstance(double,Mat)):
            return Mat(cMat.MulOperator(self.MatCapsule,double),1)
        elif(isinstance(double,int)):
            return Mat(cMat.ScalarMul(self.MatCapsule,double),1)
        else:
            return Vect(cMat.MulVectOperator(self.MatCapsule,double),1)
    def __truediv__(self,double):
        return Mat(cMat.ScalarDiv(self.MatCapsule,double),1)
    def inv(self):
        return Mat(cMat.Inverse(self.MatCapsule),1)
    def toNumpy(self): 
        return cMat.toNum(self.MatCapsule)
    #def __mul__(self,Ma):
     #   return Mat(cMat.MulOperator(self.MatCapsule,Ma),1)

    def __delete__(self):
        cMat.delete_object(self.MatCapsule)


