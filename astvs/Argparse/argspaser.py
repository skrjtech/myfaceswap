#!/bin/python3
# -*- coding: utf-8 -*-

import textwrap
import argparse
from argparse import (
    RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
)
from pprint import pprint

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
    parser.add_argument('--input', type=str, default='result/TrainDatas')
    parser.add_argument('--result', type=str, default='result')
    parser.add_argument('--weights-load', dest='weights_load', action='store_true')
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--max-frames', type=int, default=50)
    parser.add_argument('--identity-rate', type=float, default=5.)
    parser.add_argument('--gan-rate', type=float, default=5.)
    parser.add_argument('--recycle-rate', type=float, default=10.)
    parser.add_argument('--current-rate', type=float, default=10.)
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--start-iter', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0002)
    parser.add_argument('--decay', type=int, default=200)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--beta1', type=int, default=0.5)
    parser.add_argument('--beta2', type=int, default=0.999)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--skip', type=int, default=2)

    return parser.parse_args()

def EvalArgs():

    parser = argparse.ArgumentParser(description='tl-fsm training Script')
    parser.add_argument('--input', type=str, default='0')
    parser.add_argument('--result', type=str, default='/result')
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--cuda', dest='cuda', action='store_true')

    return parser.parse_args()

def RealtimeArgs():
    return

class ArgsHelpFormat(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    pass

def argumentParser():
    parse = argparse.ArgumentParser(
        description=textwrap.dedent("""
        姿入換翻訳ビデオ会議システム
        """),
        formatter_class=ArgsHelpFormat,
        epilog=textwrap.dedent("""
        """)
    )

    subparsers = parse.add_subparsers()

    makedata = subparsers.add_parser(
        name='makedata',
        formatter_class=ArgsHelpFormat,
        help=textwrap.dedent(f"""
        ( {'makedata':<9s}) 学習用動画ファイルの構築
            -a, --domain-a              ドメインAをBに変更 (default: False)
            -b, --domain-b              ドメインBをAに変更 (default: False)
            -m, --multiframes           フレーム数を分割にまとめ処理 (default: False)
            --batch-size BATCH_SIZE     バッチサイズで処理 (default: 4)
            --cuda                      GPU上で処理 (default: False)
            -i INPUT, --input INPUT     動画ファイル先 (default: Datas/Videos)
            -o OUTPUT, --output OUTPUT  処理済みデータ出力先 (default: Datas/TrainData)
        """)
    )
    makedata.add_argument(
        '-a', '--domain-a', action='store_true', help='ドメインAをBに変更'
    )
    makedata.add_argument(
        '-b', '--domain-b', action='store_true', help='ドメインBをAに変更'
    )
    makedata.add_argument(
        '-m', '--multiframes', action='store_true', help='フレーム数を分割にまとめ処理'
    )
    makedata.add_argument(
        '--batch-size', type=int, default=4, help='バッチサイズで処理'
    )
    makedata.add_argument(
        '--cuda', action='store_true', help='GPU上で処理'
    )
    makedata.add_argument(
        '-i', '--input', type=str, default='Datas/Videos', help='動画ファイル先'
    )
    makedata.add_argument(
        '-o', '--output', type=str, default='Datas/TrainData', help='処理済みデータ出力先'
    )
    makedata.set_defaults(handler=lambda x: pprint(vars(x)))


    training = subparsers.add_parser(
        name='train',
        formatter_class=ArgsHelpFormat,
        help=textwrap.dedent(f"""
        ( {'train':<9s}) 学習用メソッド
            -i INPUT, --input INPUT                     ファイル先 (default: Datas/train/video)
            -r RESULT, --result RESULT                  学習経過重み保存先 (default: Datas/result)
            --channels CHANNELS                         画像チャンネル数 (default: 3)
            --weight                                    重み呼び出し (default: False)
            --skip-frames SKIP_FRAMES                   学習用データ間スキップ数 (default: 2)
            --max-frames MAX_FRAMES                     学習時比較用保存フレーム数 (default: 50)
            -epochs EPOCHS                              データ全体の学習回数 (default: 50)
            --epoch-start EPOCH_START                   学習回数開始位置 (default: 1)
            --epoch-decay EPOCH_DECAY                   学習回数減衰数 (default: 200)
            --batch-size BATCH_SIZE                     バッチサイズで処理 (default: 4)
            --lr LR                                     学習率 (default: 0.0002)
            --beta1 BETA1                               ベータ開始値 (default: 0.9)
            --beta2 BETA2                               ベータ終了値 (default: 0.999)
            --cuda                                      GPU上で処理 (default: False)
            --interval INTERVAL                         結果表示・重み保存インターバル (default: 10)
            --identity-loss-rate IDENTITY_LOSS_RATE     IdentityLoss 調整値 (default: 5.0)
            --gan-loss-rate GAN_LOSS_RATE GanLoss       調整値 (default: 5.0)
            --recycle-loss-rate RECYCLE_LOSS_RATE       RecycleLoss 調整値 (default: 10.0)
            --recurrent-loss-rate RECURRENT_LOSS_RATE   RecurrentLoss 調整値 (default: 10.0)
        """)
    )
    training.add_argument(
        '-i', '--input', type=str, default='Datas/train/video', help='ファイル先'
    )
    training.add_argument(
        '-r', '--result', type=str, default='Datas/result', help='学習経過重み保存先'
    )
    training.add_argument(
        '--channels', type=int, default=3, help='画像チャンネル数'
    )
    training.add_argument(
        '--weight', action='store_true', help='重み呼び出し'
    )
    training.add_argument(
        '--skip-frames', type=int, default=2, help='学習用データ間スキップ数'
    )
    training.add_argument(
        '--max-frames', type=int, default=50, help='学習時比較用保存フレーム数'
    )
    training.add_argument(
        '-epochs', type=int, default=50, help='データ全体の学習回数'
    )
    training.add_argument(
        '--epoch-start', type=int, default=1, help='学習回数開始位置'
    )
    training.add_argument(
        '--epoch-decay', type=int, default=200, help='学習回数減衰数'
    )
    training.add_argument(
        '--batch-size', type=int, default=4, help='バッチサイズで処理'
    )
    training.add_argument(
        '--lr', type=float, default=0.0002, help='学習率'
    )
    training.add_argument(
        '--beta1', type=float, default=0.9, help='ベータ開始値'
    )
    training.add_argument(
        '--beta2', type=float, default=0.999, help='ベータ終了値'
    )
    training.add_argument(
        '--cuda', action='store_true', help='GPU上で処理'
    )
    training.add_argument(
        '--interval', type=int, default=10, help='結果表示・重み保存インターバル'
    )
    training.add_argument(
        '--identity-loss-rate', type=float, default=5.0, help='IdentityLoss 調整値'
    )
    training.add_argument(
        '--gan-loss-rate', type=float, default=5.0, help='GanLoss 調整値'
    )
    training.add_argument(
        '--recycle-loss-rate', type=float, default=10.0, help='RecycleLoss 調整値'
    )
    training.add_argument(
        '--recurrent-loss-rate', type=float, default=10.0, help='RecurrentLoss 調整値'
    )
    training.set_defaults(handler=lambda x: pprint(vars(x)))

    eval = subparsers.add_parser(
        name='eval',
        formatter_class=ArgsHelpFormat,
        help=textwrap.dedent(f"""
        ( {'eval':<9s}) 評価用メソッド
            -a, --domain-a              ドメインAをBに変更 (default: False)
            -b, --domain-b              ドメインBをAに変更 (default: False)
            -i IMAGE, --image IMAGE     評価用画像ファイル先 (default: )
            -r RESULT, --result RESULT  重み保存先 (default: Datas/result)
            -o OUTPUT, --output OUTPUT  評価済みデータ保存先 (default: Datas/result/eval)
            -v VIDEO, --video VIDEO     評価用動画ファイル先 (default: )
            --cuda                      GPU上で処理 (default: False)
        """)
    )
    eval.add_argument(
        '-a', '--domain-a', action='store_true', help='ドメインAをBに変更'
    )
    eval.add_argument(
        '-b', '--domain-b', action='store_true', help='ドメインBをAに変更'
    )
    eval.add_argument(
        '-i', '--image', type=str, default='', help='評価用画像ファイル先'
    )
    eval.add_argument(
        '-r', '--result', type=str, default='Datas/result', help='重み保存先'
    )
    eval.add_argument(
        '-o', '--output', type=str, default='Datas/result/eval', help='評価済みデータ保存先'
    )
    eval.add_argument(
        '-v', '--video', type=str, default='', help='評価用動画ファイル先'
    )
    eval.add_argument(
        '--cuda', action='store_true', help='GPU上で処理'
    )
    
    eval.set_defaults(handler=lambda x: pprint(vars(x)))


    plugin = subparsers.add_parser(
        name='plugin',
        formatter_class=ArgsHelpFormat,
        help=textwrap.dedent(f"""
        ( {'plugin':<9s}) リアルタイムメソッド
            -a, --domain-a              ドメインAをBに変更 (default: True)
            -b, --domain-b              ドメインBをAに変更 (default: False)
            -d DEVICE, --device DEVICE  仮想カメラ用ファイル名 (default: -1)
            -n NAME, --name NAME        キャプチャ名 (default: ASTVS)
            -r RESULT, --result RESULT  重み保存先 (default: Datas/result)
            -v VIDEO, --video VIDEO     カメラ認識番号 (default: 0)
            --cuda                      GPU上で処理 (default: False)
        """)
    )
    plugin.add_argument(
        '-a', '--domain-a', action='store_false', help='ドメインAをBに変更'
    )
    plugin.add_argument(
        '-b', '--domain-b', action='store_true', help='ドメインBをAに変更'
    )
    plugin.add_argument(
        '-d', '--device', type=str, default='-1', help='仮想カメラ用ファイル名'
    )
    plugin.add_argument(
        '-n', '--name', type=str, default='ASTVS', help='キャプチャ名'
    )
    plugin.add_argument(
        '-r', '--result', type=str, default='Datas/result', help='重み保存先'
    )
    plugin.add_argument(
        '-v', '--video', type=str, default='0', help='カメラ認識番号'
    )
    plugin.add_argument(
        '--cuda', action='store_true', help='GPU上で処理'
    )
    def Switch(args):
        if args.domain_b:
            args.domain_a = not args.domain_a
        pprint(vars(args))
    plugin.set_defaults(handler=lambda x: Switch(x))

    args = parse.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parse.print_help()

if __name__ == '__main__':
    argumentParser()