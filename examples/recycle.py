import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from myfaceswap.preprocessing.mkdata import CleanBack
from myfaceswap.models.recyclemodel import RecycleModel

Source = "/Output/Database/source.mp4"
Target = "/Output/Database/target.mp4"
ioRoot = "/Output/ioRoot"

CleanBack(rootDir=ioRoot, domains=[Source, Target], batchSize=4, limit=-1, gpu=True).video2frame()

RecycleModel(
    rootDir=ioRoot,
    source=os.path.join(ioRoot, "Datasets/source/video"),
    target=os.path.join(ioRoot, "Datasets/target/video"),
    batchSize=1,
    gpu=True,
    cpuWorkers=4,
    learningRate=0.0002,
    epochs=50
).Train()