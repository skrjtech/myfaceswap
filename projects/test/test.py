import cv2
import numpy
from functools import singledispatch

class Sample(object):
    def __init__(self, index) -> None:
        self.__build__(index)

    @singledispatch
    def __build__(self, variable: int):
        self.variable = variable
    
    @__build__.register
    def _(self, variable: str):
        self.variable = variable
    
    @__build__.register
    def _(self, variable: list):
        self.variable = variable

    def __str__(self) -> str:
        return f"result: {self.variable}"

if __name__ == '__main__':
    print(Sample('string'))
    print(Sample(1))
    print(Sample([1, 2, 3]))    

    