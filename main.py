import json
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from data import Dataset
from model import Generator, Discriminator
from run import run
from utils import save_loss, save_img

parser = argparse.ArgumentParser()
parser.add_argument('dataset',
                    help="Select the dataset from ['food-101', 'food-101-small']")
args = parser.parse_args()

if __name__ == '__main__':
    config_dir = os.path.join(os.getcwd(),
                              'configs/main.json')
    with open(config_dir, 'r') as f:
        config = json.load(f)
    
    torch.manual_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    Dataset = Dataset(args.dataset)
    dl = Dataset.get_dataloader()
    config_data = Dataset.get_config()
    config['data'] = config_data

    Generator     = Generator(config).to(config['device'])
    Discriminator = Discriminator(config).to(config['device'])
    criterion = nn.BCEWithLogitsLoss()
    optim_g = optim.Adam(Generator.parameters(),
                         lr=config['optim']['lr_g'],
                         betas=(config['optim']['betas1_g'],
                                config['optim']['betas2_g']))
    optim_d = optim.Adam(Discriminator.parameters(),
                         lr=config['optim']['lr_d'],
                         betas=(config['optim']['betas1_d'],
                                config['optim']['betas2_d']))
    
    losses_d = []
    losses_g = []
    for epoch in range(config['epochs']):
        loss_d, loss_g = run(config,
                             Generator,
                             Discriminator,
                             dl,
                             criterion,
                             optim_g,
                             optim_d,
                             state='train')

        print(f"Epoch: {epoch+1}, Loss_D: {loss_d:.4f}, Loss_G: {loss_g:.4f}",
                flush=True)
        losses_d.append(loss_d)
        losses_g.append(loss_g)
    
    save_loss(losses_d, losses_g, args.dataset)
    save_img(config, Generator, args.dataset)
    

        
