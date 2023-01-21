#!/bin/python3
# -*- coding: utf-8 -*-

import os
from Argparse import TrainArgs
from ReCycleGan import RecycleTrain
def main():
    args = TrainArgs()
    RecycleTrain(args).Train()

if __name__ == '__main__':
    main()