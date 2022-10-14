from email.policy import strict
import os 
import torch
import mydeepface.utils as myutils
from mydeepface.datasetGenerator import FaseDatasetSquence
from mydeepface.models import Generator, Discriminator, Predictor
from argumentparce import Args, argumentparse
def train():
    parse = argumentparse()

    G_A2B = Generator(parse.input_channel, parse.output_channel)
    G_B2A = Generator(parse.output_channel, parse.input_channel)
    
    D_A = Discriminator(parse.input_channel)
    D_B = Discriminator(parse.output_channel)

    P_A = Predictor(parse.input_channel * parse.input_channel)
    P_B = Predictor(parse.output_channel * parse.output_channel)

    if parse.gpu:
        map(lambda x: x.cuda(), [G_A2B, G_B2A, D_A, D_B, P_A, P_B])
    
    map(lambda x: x.apply(myutils.weights_init_normal), [G_A2B, G_B2A, D_A, D_B])

    if parse.load_model_params:
        map(
            lambda x, y: x.load_state_dict(torch.load(os.path.join(parse.model_params_path, y), map_location="cuda:0"), strict=False),
            [(val, key) for key, val in {"G_A2B": G_A2B, "G_B2A": G_B2A, "D_A": D_A, "D_B": D_B, "P_A": P_A, "P_B": P_B}]
        )
    
