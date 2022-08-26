from omegaconf import OmegaConf
import yaml
from models.discriminator import Discriminator
from models.generator import Generator
from train.utils import weights_init,get_noise
import torch 
import os
from tqdm import tqdm
import numpy as np
import wandb
from data.data import DataloaderDCGAN,get_label
from torchvision.utils import save_image

# Initialise project on W&B
wandb.init(project='From_GAN_to_DCGAN',name='Génération_01_lr_0.0002_epoch_30')

def training(config ='./configs/config.yaml'):

    with open(config, 'r') as f:
        configuration = OmegaConf.create(yaml.safe_load(f))
    
    #check if GPUs are available(to make training faster)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # define our discriminator and generator 
    # Inialise their weights (as mentionned in the paper)
    gen = Generator(configuration.train.noise_dim)
    gen.apply(weights_init)
    gen.to(device)
    disc = Discriminator (configuration.train.img_shape)
    disc.apply(weights_init)
    disc.to(device)


    # define optimizers
    optimizer_gen  = torch.optim.Adam(gen.parameters(),configuration.optimizer.lrG,(0.5,0.999))
    optimizer_disc = torch.optim.Adam(disc.parameters(),configuration.optimizer.lrD,(0.5,0.999))


    # define loss function
    loss = torch.nn.BCELoss()

    # get data for training
    dataset=  DataloaderDCGAN(configuration.data.img_path,configuration.train.img_shape)
    dataloader = torch.utils.data.DataLoader(dataset,configuration.train.batch, shuffle=True,drop_last=True)


    # training loop
    for epoch in tqdm(range(configuration.train.epochs)):
        g_loss =[]
        d_loss=[]
        for indx,batch in enumerate(dataloader):

            # create labels
            real_label,fake_label = get_label(configuration.train.batch)
            real_label=real_label.to(device)
            fake_label = fake_label.to(device)

                                                ###train the generator network####
            # make sure all grads are zero
            optimizer_gen.zero_grad()

            # create generator_input and generate output
            noise = get_noise(configuration.train.batch,configuration.train.noise_dim).to(device)
            fake_images = gen(noise)
            
            # classify generator output and resize it as the
            disc_fake_output = disc(fake_images).view(real_label.size()[0],-1)

            # Calculate generator_loss
            generator_loss = loss(disc_fake_output,real_label)

            g_loss.append(generator_loss.item())

            # Compute  backward pass
            generator_loss.backward()
            #print(f"z grad-->{noise.grad}")
            optimizer_gen.step()

            
                                                ###train the discriminator network####

            # make sure all grads are zero
            optimizer_disc.zero_grad()

           # Calculate loss for real and fake images          
            disc_fake_output = disc(fake_images.detach()).view(real_label.size()[0],-1)
            fake_loss = loss(disc_fake_output,fake_label) 

            disc_real_output = disc(batch.to(device)).view(-1,1)
            real_loss = loss(disc_real_output,real_label)
            disc_loss = real_loss + fake_loss
            d_loss.append(disc_loss.item())

            # Compute  backward pass
            disc_loss.backward()
            optimizer_disc.step()

        
        # print  and track : epoch + losses
        print(f"epoch : {epoch} generator_loss :{np.mean(g_loss)} discriminator_loss : {np.mean(d_loss)} ")
        wandb.log({'train_gen_loss':np.mean(g_loss),'custom_step': epoch})
        wandb.log({'train_disc_loss':np.mean(d_loss),'custom_step': epoch})

        # save a the generator output after each epoch to visualise the evolution of the training
        path = os.path.join('./',configuration.data.output_path)
        if not os.path.exists(path):
            os.makedirs(path)
        save_image(fake_images,os.path.join(path,f"img_{epoch}.jpg"))
    

    # save laste model state
    torch.save(gen.state_dict(), os.join('./',configuration.checkpoint_path,'generator.ckpt'))


if __name__ =='__main__': 
    training()















            














