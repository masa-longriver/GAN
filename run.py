import torch

def run(config, 
        generator, 
        discriminator, 
        dl,
        criterion,
        optim_g,
        optim_d,
        state='train'):
    
    real_label = 1
    fake_label = 0
    running_loss_d = 0
    running_loss_g = 0

    if state == 'train':
        generator.train()
        discriminator.train()
    elif state == 'eval':
         generator.eval()
         discriminator.eval()

    for data, _ in dl:
        """ step1. Discriminatorの学習 """
        if state == 'train':
            discriminator.zero_grad()

        # Real Dataに対するlossを計算
        real_data = data.to(config['device'])
        label = torch.full((real_data.size(0),),
                            real_label,
                            dtype=real_data.dtype,
                            device=config['device'])
        output = discriminator(real_data).view(-1)
        loss_real = criterion(output, label)
        if state == 'train':
            loss_real.backward()

        # Fake Dataに対するlossを計算
        noise = torch.randn(real_data.size(0), 
                            config['model']['noise_channel'], 
                            1, 
                            1,
                            device=config['device'])
        fake_data = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake_data.detach()).view(-1)
        loss_fake = criterion(output, label)
        if state == 'train':
            loss_fake.backward()

        # Discriminatorの更新
        loss_d = loss_real + loss_fake
        running_loss_d += loss_d.item() * real_data.size(0)
        if state == 'train':
            optim_d.step()

        """ step2. Generatorの学習 """
        if state == 'train':
            generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fake_data).view(-1)
        loss_g = criterion(output, label)
        running_loss_g += loss_g.item() * real_data.size(0)
        if state == 'train':
            loss_g.backward()
            optim_g.step()
    
    return running_loss_d, running_loss_g