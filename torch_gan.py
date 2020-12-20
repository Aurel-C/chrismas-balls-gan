import torch
from torch import nn,optim
import torchvision
from tqdm import trange
from torchvision import transforms
import numpy as np
import wandb

def train(data,epochs):
    data = np.transpose(data,[0,3,1,2]).astype(np.float32)
    dataloader = torch.utils.data.DataLoader(data,batch_size=32,shuffle=True,drop_last=True)
    n_batches = len(dataloader)
    device = torch.device("cuda:0")
    preprocess = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize([193,172,167],[71.1,86.3,88.3]),
    ])
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    loss = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr= 0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr= 0.0002, betas=(0.5, 0.999))
    wandb.init()
    loss_dict = {
        "g_loss":0,
        "d_loss":0,
    }

    for epoch in trange(epochs):
        for data in dataloader:
            data = preprocess(data.to(device))
            g_labels = torch.ones(data.size(0),device=device)
            fake_labels = torch.zeros(data.size(0),device=device)
            true_labels = torch.ones(data.size(0),device=device)

            inputs = torch.randn(data.size(0),100,1,1,device=device)
            generated = netG(inputs)


            optimizerD.zero_grad()

            fake_pred = netD(generated.detach()).view(-1)
            fake_loss = loss(fake_pred,fake_labels)
            fake_loss.backward()

            true_pred = netD(data).view(-1)
            true_loss = loss(true_pred,true_labels)
            true_loss.backward()

            optimizerD.step()


            optimizerG.zero_grad()
            pred = netD(generated).view(-1)
            g_loss = loss(pred,g_labels)
            g_loss.backward()
            optimizerG.step()


            loss_dict["g_loss"]+= g_loss.mean()
            loss_dict["d_loss"]+= (fake_loss + true_loss).mean()

        loss_dict["g_loss"]/= n_batches
        loss_dict["d_loss"]/= n_batches
        wandb.log(loss_dict) 
        loss_dict["g_loss"] = 0
        loss_dict["d_loss"] = 0

        if epoch % 10 == 0:
            wandb.log({"generated":[wandb.Image(generated[i]) for i in range(generated.size(0))]})

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (64*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (64*4) x 8 x 8
            nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (64*2) x 16 x 16
            nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            # state size. (3) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)
