#!/bin/python3
# -*- coding: utf-8 -*-

import textwrap
import argparse
from argparse import (
    RawTextHelpFormatter, 
    ArgumentDefaultsHelpFormatter)
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
    
    # 作業スペースの確保　(Init)
    init_ws = subparsers.add_parser(
        name='init',
        formatter_class=ArgsHelpFormat,
        help=textwrap.dedent(f"""
        ( {'init':<9s}) ワークスペースの作成
        """)
    )

    # 訓練・評価用データセット作成オプション
    makedata = subparsers.add_parser(
        name='makedata',
        formatter_class=ArgsHelpFormat,
        help=textwrap.dedent(f"""
        ( {'makedata':<9s}) 学習用動画ファイルの構築
            -m, --multiframes           フレーム数を分割にまとめ処理 (default: False)
            --batch-size BATCH_SIZE     バッチサイズで処理 (default: 4)
            --cuda                      GPU上で処理 (default: False)
        """)
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

    training = subparsers.add_parser(
        name='train',
        formatter_class=ArgsHelpFormat,
        help=textwrap.dedent(f"""
        ( {'train':<9s}) 学習用メソッド
            --channels CHANNELS                         画像チャンネル数 (default: 3)
            --weights                                   重み呼び出し (default: False)
            --skip-frames SKIP_FRAMES                   学習用データ間スキップ数 (default: 2)
            --max-frames MAX_FRAMES                     学習時比較用保存フレーム数 (default: 50)
            --epochs EPOCHS                              データ全体の学習回数 (default: 50)
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
        '--channels', type=int, default=3, help='画像チャンネル数'
    )
    training.add_argument(
        '--weights', action='store_true', help='重み呼び出し'
    )
    training.add_argument(
        '--skip-frames', type=int, default=2, help='学習用データ間スキップ数'
    )
    training.add_argument(
        '--max-frames', type=int, default=50, help='学習時比較用保存フレーム数'
    )
    training.add_argument(
        '--epochs', type=int, default=50, help='データ全体の学習回数'
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

    eval = subparsers.add_parser(
        name='eval',
        formatter_class=ArgsHelpFormat,
        help=textwrap.dedent(f"""
        ( {'eval':<9s}) 評価用メソッド
            -i IMAGE, --image IMAGE     評価用画像ファイル先 (default: )
            -o OUTPUT, --output OUTPUT  評価済みデータ保存先 (default: )
            -v VIDEO, --video VIDEO     評価用動画ファイル先 (default: )
            --cuda                      GPU上で処理 (default: False)
        """)
    )
    eval.add_argument(
        '-i', '--image', type=str, default='', help='評価用画像ファイル先'
    )
    eval.add_argument(
        '-v', '--video', type=str, default='', help='評価用動画ファイル先'
    )
    eval.add_argument(
        '-o', '--output', type=str, default='', help='評価済みデータ保存先'
    )
    eval.add_argument(
        '--cuda', action='store_true', help='GPU上で処理'
    )

    plugin = subparsers.add_parser(
        name='plugin',
        formatter_class=ArgsHelpFormat,
        help=textwrap.dedent(f"""
        ( {'plugin':<9s}) リアルタイムメソッド
            -a, --domain-a              ドメインAをBに変更 (default: True)
            -b, --domain-b              ドメインBをAに変更 (default: False)
            -d DEVICE, --device DEVICE  仮想カメラ用ファイル名 (default: -1)
            -n NAME, --name NAME        キャプチャ名 (default: ASTVS)
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
        '-v', '--video', type=int, default=0, help='カメラ認識番号'
    )
    plugin.add_argument(
        '--cuda', action='store_true', help='GPU上で処理'
    )
    plugin.add_argument(
        '--mic', type=str, default=0, help='マイク認識番号'
    )

    return parse, init_ws, makedata, training, eval, plugin

if __name__ == '__main__':
    argumentParser()