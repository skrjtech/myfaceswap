#!/bin/python3
# -*- coding: utf-8 -*-

import os

import cv2

from Argparse import EvalArgs, TrainArgs
from ReCycleGan import RecycleEval
from CameraAndAudio import CameraPlugin

def main():

    args = TrainArgs()
    REval = RecycleEval(args)
    REval.Build()
    camPlugin = CameraPlugin(args.input)

    while camPlugin.isOpened():
        frame = camPlugin.Plugin()
        frame = cv2.resize(frame, (256, 256))
        frame = REval.Plugin(frame)
        frame = cv2.resize(frame, (640, 460))
        camPlugin.imshow(frame)

if __name__ == '__main__':
    main()