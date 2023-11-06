import datetime as dt
import os
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

def make_dir(path):
    path_list = path.split('/')
    now_path = ""
    for i, dir in enumerate(path_list):
       if i == 0:
          continue
       else:
          now_path += f"/{dir}"
          if not os.path.exists(now_path):
             os.makedirs(now_path)

def save_loss(loss_d, loss_g, dataset):
    path = os.path.join(os.getcwd(), 'log', 'losses', dataset.lower())
    make_dir(path)

    plt.figure()
    plt.figure(figsize=(10, 5))
    plt.plot(loss_d, label='Discriminator')
    plt.plot(loss_g, label='Generator')
    plt.title("losses")
    plt.ylabel('loss')
    plt.xlabel("epoch")
    plt.legend()

    now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_nm = os.path.join(path, f'{now}_loss.png')
    plt.savefig(file_nm)
    plt.close()

def save_img(config, generator, dataset):
    path = os.path.join(os.getcwd(), 'log', 'img', dataset.lower())
    make_dir(path)

    with torch.no_grad():
        noise = torch.randn(12,
                            config['model']['noise_channel'],
                            1,
                            1,
                            device=config['device'])
        generate_img = generator(noise).detach().cpu()
        grid = vutils.make_grid(generate_img,
                                nrow=4,
                                padding=2,
                                normalize=True)
        plt.axis('off')
        plt.imshow(grid.permute(1, 2, 0))
        now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_nm = os.path.join(path, f'{now}_generate_img.png')
        plt.savefig(os.path.join(path, file_nm), bbox_inches='tight')
        plt.close()