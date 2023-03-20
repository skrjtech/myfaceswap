#!/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
from Argparse import MkDataArgs
from CleanBack import VideoFileCleanBackModel

def main():
    args = MkDataArgs()
    print(args)

    # Check Input
    if args.input == '':
        print('no input directory specified.')
        sys.exit(0)
    if not os.path.exists(args.input):
        print(f"don't find {args.input}.")
        sys.exit(0)

    # Check Output
    if args.output == '':
        print('no output directory specified.')
        sys.exit(0)
    if not os.path.exists(args.output):
        print(f"don't find {args.output}.")
        print("creating new output directory.")
        os.makedirs(args.output, exist_ok=True)

    # Check Domain
    if args.domainA == args.domainB:
        print('please, select domain A or B.')
        print('you are selecting domains A and B.')
        sys.exit(0)

    Domain = None
    if args.domainA:
        Domain = 'DomainA'
    elif args.domainB:
        Domain = 'DomainB'

    InputPath = args.input
    OutputPath = os.path.join(args.output, Domain)

    # Cuda
    Cuda = None
    if args.cuda:
        Cuda = 'cuda'

    # View Infos
    View = f"""
    -------------------------
    Input Path: {InputPath}
    Output Path: {OutputPath}
    Selecting Domain: {Domain}
    Batch Size: {args.batch_size}
    Use Cuda: {Cuda}
    -------------------------
    """
    print(View)
    for exec in ["*.mp4", "*.MP4", "*.mov"]:
        PathList = glob.glob(os.path.join(InputPath, exec))
        if len(PathList) > 0:
            break
    for inputPath in PathList:
        videoFileName = inputPath.split('/')[-1].split('.')[0]
        OutpuDomainPath = os.path.join(OutputPath, videoFileName)
        if not os.path.exists(OutpuDomainPath):
            os.makedirs(OutpuDomainPath, exist_ok=True)
        VideoFileCleanBackModel(
            camIdx=inputPath,
            outputPath=OutpuDomainPath,
            device=Cuda,
            batchSize=args.batch_size
        ).Run()

if __name__ == '__main__':
    main()