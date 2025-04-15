import torch
import yaml
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from DDPM import *
import torch.optim as optim
import numpy
import os

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--mode', type=str, required=True, help='training or inference')
    args = parser.parse_args()
    return args





def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace




def training(config, device):

# ---------------------------         Load the dataloader        --------------------------------------

    # Root directory for the dataset
    data_root = config.data.data_root
    # Spatial size of training images, images are resized to this size.
    image_size = config.data.image_size

    if config.training.dataset == "CelebA":
        dataset = torchvision.datasets.CelebA(data_root,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                            std=[0.5, 0.5, 0.5])
                                    ]))

    if config.training.dataset == "MNIST":
        dataset = torchvision.datasets.MNIST(data_root,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5],
                                                            std=[0.5])
                                    ]))

    #  ----------------------------           Sampling  and Training    -----------------------------------------------------

    dataloader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True)

    unet = UNet(config).to(device)
    LNS = LinearNoiseScheduler(config.LinearNoiseScheduler.time_steps, config.LinearNoiseScheduler.beta_start, config.LinearNoiseScheduler.beta_end, device)

    lr = config.training.lr
    optimizer = optim.Adam(unet.parameters(), lr=lr, weight_decay=0.0)   
    # lr_schedul = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.2) #lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.2, total_iters=50)
    loss = nn.MSELoss()
    for epoch in range(566,config.training.epochs):
        total_loss = 0
        n_samples = 0

        for im,_ in dataloader:
            im = im.to(device)
            epsilon = torch.normal(0, 1, size=im.size()).to(device)
            t = torch.randint(0, config.LinearNoiseScheduler.time_steps, (im.shape[0],)).to(device)
            t_emb =  get_time_embedding(t, config.LinearNoiseScheduler.t_emb_dim)
            im_noisy = LNS.add_noise(im, epsilon, t)
            e = unet(im_noisy, t_emb)
            optimizer.zero_grad()
            l = loss(e, epsilon)
            l.backward()
            optimizer.step()
            total_loss += l.item()
            n_samples += im.size(0)

        print("epoch:", epoch, "loss:", total_loss/n_samples)
        isExist = os.path.exists(config.training.model_path)
        if not isExist:

        # Create a new directory because it does not exist
            os.makedirs(config.training.model_path)
        if epoch%5 == 0:
            torch.save(unet.state_dict(), config.training.model_path+"ckpt"+str(epoch)+".pth")   







# -----------------------------------------           Inference         --------------------------------------------------

def inference(config, device):

    # Initialize the model
    unet = UNet(config).to(device)
    LNS = LinearNoiseScheduler(config.LinearNoiseScheduler.time_steps, config.LinearNoiseScheduler.beta_start, config.LinearNoiseScheduler.beta_end, device)

    # Load pretrained weights to DDPM
    ckpt =  torch.load(config.test.model_path, weights_only=True)
    unet.load_state_dict(ckpt)
    unet.to(device)
    unet.eval()
    print("Model loaded")
    for param in unet.parameters():
                param.requires_grad = False

    # Initiate a random normal noise as epsilon
    c, h, w = 1, config.data.image_size, config.data.image_size
    batch_size = 1
    im_noisy = torch.normal(0, 1, size=(batch_size,c,h,w)).to(device)
    
    isExist = os.path.exists("./outcomes")
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(config.training.model_path)


    for t in range(999,0,-1):
        t = torch.tensor(t).unsqueeze(0)
        t = t.to(device)
        t_emb =  get_time_embedding(t, config.LinearNoiseScheduler.t_emb_dim)
        t_emb = t_emb.to(device)
        e = unet(im_noisy, t_emb)
        im_noisy, x0 = LNS.sampling_from_xt_1( im_noisy, e, t)
        

        if t%110 == 1:
            
            ims = torch.clamp(im_noisy, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2                    
            torchvision.utils.save_image(ims, "./outcome/"+str(t.item())+ ".jpg")  

        





# -----------------------------------    Main         --------------------------------------


if __name__ == '__main__':

    args = parse_args_and_config() 

    with open(args.config) as f:
        config = yaml.safe_load(f)
        config = dict2namespace(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "inference":
        test(config, device)
    if args.mode == "training":
        training(config, device)


