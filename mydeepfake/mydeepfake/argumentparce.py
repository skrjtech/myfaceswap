import argparse

class Args:
    def __init__(self, args) -> None:
        
        # devices
        self.gpu = args.gpu
        self.num_work_cpu = args.num_work_cpu
        
        # Train info
        self.start_epoch = args.start_epoch
        self.max_epochs = args.max_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.decay_ecpoh = args.decay_ecpoh
        self.load_model_params = args.load_model_params
        self.model_params_path = args.model_params_path

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
    # device
    parse.add_argument()
    # Train
    parse.add_argument()
    # Test
    parse.add_argument()
    # I/O File and Dirrectory
    parse.add_argument()

    return Args(parse.parse_args())