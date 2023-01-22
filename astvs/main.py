#!/bin/python3
# -*- coding: utf-8 -*-
import cv2
from Argparse.modelargs import *
from Argparse import argumentParser
from CameraAndAudio import CameraPlugin, AudioPlugin
from CleanBack import CleanBackModel
from CleanBack import VideoFileCleanBack
from ReCycleGan import RecycleEval
from concurrent.futures import ThreadPoolExecutor

def MakeData(args):
    args = MakedataArgs(args)
    VideoFileCleanBack(args)

def Train(args):
    args = TrainArgs(args)

def Eval(args):
    args = EvalArgs(args)

def Realtime(args):
    args = PluginArgs(args)
    Camera = CameraPlugin(args.VIDEO)
    # Mic = AudioPlugin(args.MIC)
    CBM = CleanBackModel(args)
    Recycle = RecycleEval(args)
    # WhisperTans =
    Datas = [None, None]
    def VideoRun(*a):
        while True:
            frame = Camera.Read()
            cleanBack = CBM(frame)
            output = Recycle(cleanBack)
            Datas[0] = output
    # def AudioRun(*a):
    #     while True:
    #         audio = Mic.Read()
    #         output = WhisperTans(audio)
    #         Datas[1] = output
    def ProcessRun(*a):
        while True:
            frame = Datas[0]
            if Datas[1] is not None:
                frame = cv2.putText(frame, Datas[0], (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1, cv2.LINE_AA)
            Camera.imshow(frame)
    try:
        with ThreadPoolExecutor(max_workers=8) as exec:
            exec.map(VideoRun, [None])
            # exec.map(AudioRun, [None])
            exec.map(ProcessRun, [None])
            
    except KeyboardInterrupt():
        print("実行終了します．")

def main():
    parse, makedata, training, eval, plugin = argumentParser()
    makedata.set_defaults(handler=MakeData)
    training.set_defaults(handler=Train)
    eval.set_defaults(handler=Eval)
    plugin.set_defaults(handler=Realtime)

    args = parse.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parse.print_help()

if __name__ == '__main__':
    main()