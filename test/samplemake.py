import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from mydeepfake.processing import Video2FramesAndCleanBack
Video2FramesAndCleanBack("io_root", "base1.mp4", "base1.mp4", 4, True, True, 1)()