import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.simplefilter('ignore')
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from myfaceswap.models import RecycleTrainer
trainer = RecycleTrainer('/ws/ioroot', '/ws/Datasets/domainA/video', '/ws/Datasets/domainB/video', 'epoch_00000.pth', epochs=3, gpu=True, workersCpu=2)
# trainer.ModelTest()
trainer.train()