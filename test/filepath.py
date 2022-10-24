import os 
import sys
print(__file__)
print(sys.path)
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
print(sys.path)