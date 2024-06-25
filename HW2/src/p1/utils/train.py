import torch
from tqdm.auto import tqdm
from utils.helper import sample_noise, save_checkpoint
from utils.validation import validation


def train(config, opt_G, opt_D, sche_G, sche_D, mode = 'DCGAN'):
    print('#####################################')
    print('\n          Training start!!!          \n')
    print('#####################################')

    epoch = config['epoch']
    device = config['device']
    noise_dim = config['noise_dim']
    G, D = config['generator'], config['discriminator']
    save_path = config["save_path"]
    criterion = config['criterion']
    trainset_loader = config['trainset_loader']
    
    G.train()
    D.train()
    log_interval = 100
    best_score = 1000
    real_score, fake_score = 1.0, 0.0
    for ep in range(1, epoch+1):
        iteration = 0
        tqbm_bar = tqdm(trainset_loader)
        for imgs in tqbm_bar:
            real_imgs = imgs.to(device)
            
            bs = real_imgs.shape[0]
            noise = sample_noise(bs, noise_dim).to(device)

            real_label = torch.full((bs,), real_score).to(device)
            fake_label = torch.full((bs,), fake_score).to(device)
            
            #############################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            #############################################################
            opt_D.zero_grad()
            fake_imgs = G(noise)

            # Minimax Loss
            if mode == 'DCGAN':
                logits_real = D(real_imgs)
                logits_fake = D(fake_imgs)
                real_loss = criterion(logits_real, real_label)
                fake_loss = criterion(logits_fake, fake_label)
                loss_D = real_loss + fake_loss
            else: # WGAN
                gp = compute_gp(D, real_imgs, fake_imgs)
                loss_D = -torch.mean(D(real_imgs)) + torch.mean(D(fake_imgs)) + 10 * gp
            
            loss_D.backward()
            opt_D.step()
            
            ##############################################
            # (2) Update G network: maximize log(D(G(z)))
            ##############################################
            opt_G.zero_grad()

            noise = sample_noise(bs, noise_dim).to(device)
            fake_imgs = G(noise)
            logits_fake = D(fake_imgs)
            
            # Minimax Loss
            if mode == 'DCGAN':
                loss_G = criterion(logits_fake, real_label)
            else: #WGAN
                loss_G = -torch.mean(logits_fake)
            
            loss_G.backward()
            opt_G.step()

            if iteration % log_interval == 0:
                tqbm_bar.set_description('Train Epoch:{}, loss_G: {:.6f}, loss_D: {:.6f}'.format(
                    ep, loss_G.item(), loss_D.item()))
            iteration += 1

        sche_G.step()
        sche_D.step()

        face_score, fid_score = validation(config)
        if face_score >= 90 and best_score > fid_score:
            best_score = fid_score
            save_checkpoint(save_path, G)



def compute_gp(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    # Sample Epsilon from uniform distribution
    eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
    eps = eps.expand_as(real_data)

    # Interpolation between real data and fake data.
    interpolation = eps * real_data + (1 - eps) * fake_data
    #interpolation.requires_grad = True
    # get logits for interpolated images
    interp_logits = netD(interpolation)
    grad_outputs = torch.ones_like(interp_logits)


    # Compute Gradients
    gradients = torch.autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute and return Gradient Norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)