import argparse
from logging import root

class Args:
    def __init__(self, args) -> None:

        # Mode
        self.train_mode: bool = args.train_mode
        self.test_mode: bool = args.test_modes
        assert self.train_mode == self.test_mode, "train-mode か test-mode のどちらか選択してください"
        
        # devices (cpu or gpu)
        self.gpu: bool = args.gpu
        self.num_work_cpu: int = args.cpus

        # device camera (id or path)
        self.camera_index: int | str = args.camera_path
        
        # model path
        self.model_load: bool = args.load_model_params
        self.model_params_path: str = args.model_params_path

        # Train info
        self.start_epoch: int = args.start_epoch
        self.max_epochs: int = args.max_epochs
        self.batch_size: int = args.batch_size
        self.lr: float = args.lr
        self.decay_epoch: int = args.decay_epoch

        # Test info
        

        # I/O File Or Direcctory
        self.dir_root = args.dir_root
        self.file_a_dir = args.file_a_dir
        self.file_b_dir = args.file_b_dir
        self.model_save_path = args.model_save_path

        # input data info
        self.input_channel = args.input_channel
        self.output_channel = args.output_channel
        self.size = args.size

        # image frame skips
        self.frame_skip = args.frame_skip
        
        # init loss rate
        self.id_loss_rate = args.id_loss_rate
        self.gan_loss_rate = args.gan_loss_rate
        self.recy_loss_rate = args.recy_loss_rate
        self.recu_loss_rate = args.recu_loss_rate
        

def argumentparse() -> Args:
    parse = argparse.ArgumentParser(description="SwapFaceAndVoice")

    # Mode
    parse.add_argument("--train-mode", action='store_true', help="学習モード (default: False)")
    parse.add_argument("--test-mode", action='store_true', help="テストモード (default: False)")
    
    # device
    parse.add_argument("--gpu", action="store_true", help='GPU上で処理 (default: False)')
    parse.add_argument("--cpus ", type=int, default=2, help='データセット取り出しに用いるCPUの数 (default: 2)')
    parse.add_argument("--camera-path", type=str, default='index:0', help='自前のカメラの選択が可能 (args: --camera-path {"src" or "index:camera-number"})')

    # Train
    parse.add_argument('--max-epochs', type=int, default=1, help='指定の全データの学習回数 (default: 1)')
    parse.add_argument('--start-epoch', type=int, default=0, help='n回数から学習の開始 (default: 0)')
    parse.add_argument('--batch-size', type=int, default=1, help='バッチ数 (default: 1)')
    parse.add_argument('--lr', type=float, default=0.001, help='学習率 (default: 0.001)')
    parse.add_argument('--beta1', type=float, default=.5, help=' (default: 0.5)')
    parse.add_argument('--beta2', type=float, default=.999, help=' (default: 0.999)')
    parse.add_argument('--decay_epoch', type=int, default=1, help='スケジューラーの減衰パラメータ (default: 1)')
    parse.add_argument('--model-load', action='store_true', help=' (default: False)')
    
    # Test
    
    # I/O File and Dirrectory | model paths
    """
    / -> is Dir
    - -> id File
    /root
        /model_params
            -model1
                .
                .
            -modelN
        /dataset
            /train
                .
                .
            /test
                .
                .
        /logges
            /model_hiddens
                .
                .
            /losses
                .
                .
    """
    parse.add_argument('--dir-root', type=str, default='root', help='学習結果、データセット等のパス (default: root)')
    parse.add_argument('--file-a-dir', type=str, default='file_a_dir', help='ドメインAパス (default: file_a_dir)')
    parse.add_argument('--file-b-dir', type=str, default='file_b_dir', help='ドメインBパス (default: file_b_dir)')
    parse.add_argument('--model-params-path', type=str, default='model_params', help='モデルの学習済みパラメータの保存先 (default: model_params)')

    # input data info
    parse.add_argument('--input-channel', type=int, default=3, help='入力チャンネル (default: 3)')
    parse.add_argument('--output-channel', type=int, default=3, help='出力チャンネル (default: 3)')
    parse.add_argument('--image-size', type=int, default=255, help='イメージサイズ (default: 255)')
    parse.add_argument('--frame-skip', type=int, default=2, help='フレームスキップ (default: 2)')

    # init loss rate
    parse.add_argument('--id-loss-rate', type=float, default=5., help=' (default: 5.0)')
    parse.add_argument('--gan-loss-rate', type=float, default=5., help=' (default: 5.0)')
    parse.add_argument('--recy-loss-rate', type=float, default=10., help=' (default: 10.0)')
    parse.add_argument('--recu-loss-rate', type=float, default=10., help=' (default: 10.0)')

    parse.parse_args()
    # return Args(parse.parse_args())

if __name__ == '__main__':
    argumentparse()