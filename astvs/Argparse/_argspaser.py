import argparse

def MkDataArgs():

    parser = argparse.ArgumentParser(description='tl-fsm creating data Script')

    parser.add_argument('--input', type=str, default='', help='input video')
    parser.add_argument('--output', type=str, default='', help='output video')
    parser.add_argument('--domainA', dest='domainA', action='store_true')
    parser.add_argument('--domainB', dest='domainB', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch Size')
    parser.add_argument('--cuda', dest='cuda', action='store_true')

    return parser.parse_args()

def TrainArgs():

    parser = argparse.ArgumentParser(description='tl-fsm training Script')

    parser.add_argument('--input', type=str, default='', help='input video')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch Size')
    parser.add_argument('--cuda', dest='cuda', action='store_true')

    return parser.parse_args()

def EvalArgs():
    return

def RealtimeArgs():
    return
