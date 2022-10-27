import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mydeepfake.utils

mydeepfake.utils.ImageProcessBackgraund(
    'TestRoot', ['./sample1.mp4']
)()