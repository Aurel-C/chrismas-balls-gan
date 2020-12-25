import gc
import torch
from torch import nn,optim
from tqdm import trange
from torchvision import transforms
import torchvision
import numpy as np
import wandb
from PIL import Image


def train(data,epochs):
    data = np.transpose(data,[0,3,1,2]).astype(np.float32)
    dataloader = torch.utils.data.DataLoader(data,batch_size=32,shuffle=True,drop_last=True,pin_memory=True)
    n_batches = len(dataloader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preprocess = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize([127.5,127.5,127.5],[127.5,127.5,127.5]),
        transforms.Normalize([193,172,167],[71.1,86.3,88.3]),
    ])
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    loss = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr= 0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr= 0.0002, betas=(0.5, 0.999))
    wandb.init(project="Gan",tags=["torch"],entity="azeru")
    loss_dict = {
        "g_loss":0,
        "d_loss":0,
    }
    fixed_input = torch.randn(25,128,device=device).detach()

    for epoch in trange(epochs+1):
        for data in dataloader:
            data = preprocess(data.to(device))
            g_labels = torch.ones(data.size(0),device=device)
            fake_labels = torch.zeros(data.size(0),device=device)
            true_labels = 1 - 0.15*torch.rand(data.size(0),device=device)

            inputs = torch.randn(data.size(0),128,device=device)
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

        if epoch % 300 == 0:
            with torch.no_grad():
            #     generated = (netG(fixed_input).detach().permute(0,2,3,1) * 127.5 + 127.5).cpu().type(torch.uint8)
                generated = netG(fixed_input).detach().cpu()
            # wandb.log({"images/generated":[wandb.Image(generated[i].numpy()) for i in range(generated.size(0))]})
            wandb.log({"images/generated":[wandb.Image(generated[i]) for i in range(generated.size(0))]})
            gc.collect()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(128,64*8*4*4,bias=False)
        self.bn = nn.BatchNorm2d(64 * 8)
        self.relu = nn.LeakyReLU(0.2,True)
        self.up_blocks = nn.ModuleList([UpBlock(64 * 2**i,64 * 2**(i-1)) for i in range(3,0,-1)])
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.last_conv = nn.Conv2d(64,3,3,padding=1,bias=False)
        # self.tanh = nn.Tanh()

    def forward(self, input):
        x = self.linear(input).view(-1,512,4,4)
        x = self.bn(x)
        x = self.relu(x)
        for i in range(3):
            x = self.up_blocks[i](x)
        x = self.up(x)
        x = self.last_conv(x)
        # x = self.tanh(x)
        return x

class UpBlock(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size= 3,padding=1,bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size,padding=padding,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2,True),
        )
    def forward(self,input):
        return self.block(input)

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

if __name__ == "__main__":
    inp = torch.zeros(2,128)
    G = Generator()
    print(G)
    print(G.forward(inp).size())