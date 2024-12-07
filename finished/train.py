import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

# ----------------------- FIXED PARAMS -----------------------------

beta_min = 1e-4     # minimum beta
beta_max = 2e-2     # maximum beta
T = 1000            # number of time steps

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Unet(torch.nn.Module):
    # takes an input image and time, returns the score function
    def __init__(self, dataset):
        super().__init__()
        if dataset.lower() == "mnist":
            self.nch = 1
            self.pad_down = 1
            self.pad_up = 0
            self.x_shape = (1, 28, 28)
        else:
            self.nch = 3
            self.pad_down = 0
            self.pad_up = 1
            self.x_shape = (3, 32, 32)
        chs = [32, 64, 128, 256, 256]
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(self.nch, chs[0], kernel_size=3, padding=1),  # (batch, ch, 28, 28)
                torch.nn.LogSigmoid(),  # (batch, 8, 28, 28)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 14, 14)
                torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # (batch, ch, 14, 14)
                torch.nn.LogSigmoid(),  # (batch, 16, 14, 14)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 7, 7)
                torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),  # (batch, ch, 7, 7)
                torch.nn.LogSigmoid(),  # (batch, 32, 7, 7)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=self.pad_down),  # (batch, ch, 4, 4)
                torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),  # (batch, ch, 4, 4)
                torch.nn.LogSigmoid(),  # (batch, 64, 4, 4)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 2, 2)
                torch.nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),  # (batch, ch, 2, 2)
                torch.nn.LogSigmoid(),  # (batch, 64, 2, 2)
            ),
        ])
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                # input is the output of convs[4]
                torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                torch.nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=self.pad_up),  # (batch, 32, 7, 7)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                torch.nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                torch.nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], 28, 28)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                torch.nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], 28, 28)
                torch.nn.LogSigmoid(),
                torch.nn.Conv2d(chs[0], self.nch, kernel_size=3, padding=1),  # (batch, 1, 28, 28)
            ),
        ])

        self.time_embeddings = torch.nn.ModuleList([
            SinusoidalPosEmb(chs[0]),
            SinusoidalPosEmb(chs[1]),
            SinusoidalPosEmb(chs[2]),
            SinusoidalPosEmb(chs[3]),
            SinusoidalPosEmb(chs[4]),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1)
        signal = torch.reshape(x, (x.shape[0], *self.x_shape)).type(torch.float)  # (..., ch0, 28, 28)
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            signal = signal  + self.time_embeddings[i](t)[:,:,None,None]
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
            if i < len(self._convs) - 1:
                signal = signal + self.time_embeddings[-i-2](t)[:,:,None,None]
        signal = torch.reshape(signal, x.shape)  # (..., 1 * 28 * 28)
        return signal



class DDPM:
    
    def __init__(self, beta_min, beta_max, T, net, device, 
                dataset = "MNIST", 
                batchsize = 64):

        self.net = net
        self.device = device

        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.betas = torch.linspace(beta_min, beta_max, T, dtype=torch.float).to(device)
        self.alphas = (1 - self.betas).to(device)
        self.alpha_bars = torch.cumprod(self.alphas, 0).to(device)

        if dataset.lower() == "mnist":
            self.data_shape = (1, 28, 28)
        else:
            self.data_shape = (3, 32, 32)

        self.load_dataset(dataset, 64)


    def load_dataset(self, dataset, batch_size):
        
        if dataset.lower() == "mnist":
            transform = lambda x: ToTensor()(x) *2 - 1
            self.data_train = MNIST("./", train=True,  transform=transform, download=True)
            self.train_loader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True)
        
        elif dataset.lower() == "cifar10":
            transform = lambda x: ToTensor()(x) *2 - 1
            self.data_train = CIFAR10("./", train=True,  transform=transform, download=True)
            self.train_loader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True)



    def train(self, num_epochs, lr, log_interval, file_name, 
            weight_decay=0, 
            save_model_very=10,
            save_path=None,
            num_samples=64):
        t1 = time.perf_counter()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.MSELoss()

        self.net.train()

        train_loss = []

        for i in tqdm(range(num_epochs)):

            t_loss = 0
            for x, y in self.train_loader:
                x = x.to(self.device)
                eps = torch.randn_like(x).to(self.device)
                t = torch.randint(self.T, (x.shape[0],)).to(self.device)

                coeffs = self.alpha_bars[t]
                input = coeffs[:,None,None,None].sqrt()*x + (1-coeffs[:,None,None,None]).sqrt()*eps
                output = self.net(input, t)

                optimizer.zero_grad()
                loss = loss_fn(output, eps)
                loss.backward()
                optimizer.step()

                t_loss += loss.item()

            train_loss.append(t_loss / len(self.train_loader))

            if i % log_interval == 0: 
                print(f"Epoch {i}, train loss: {train_loss[-1]}")

            if i % save_model_very == 0:
                torch.save(self.net.state_dict(), save_path + "MODELS/" + f"{file_name}_EPOCH_{i}.pt")
                
                epoch_dir = save_path + f"SAMPLES/EPOCH_{i}/"
                os.mkdir(epoch_dir)
                samples = self.sample(num_samples)
                self.save_samples(samples, epoch_dir)
                del samples

        torch.save(self.net.state_dict(), save_path + file_name + "_final.pt")
        t2 = time.perf_counter()

        print(f"Finished training in : {(t2-t1)/60:.3f} minutes.\nEpochs : {i}")

    def sample(self, num_samples):

        self.net.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, *self.data_shape).to(self.device)

            for t in range(self.T,0,-1):
                if t > 1:
                    z = torch.randn_like(x).to(self.device)
                else:
                    z = torch.zeros_like(x).to(self.device)
            
                output = self.net(x, torch.tensor([t]*num_samples).to(self.device))
                
                x = (1/self.alphas[t-1].sqrt() 
                    * (x - (1-self.alphas[t-1])/(1-self.alpha_bars[t-1]).sqrt() * output) 
                    + self.betas[t-1].sqrt() * z)
        
        x = torch.clamp((x + 1) * 0.5, 0, 1)

        return x.cpu()
    

    def save_samples(self, x, sample_path):
        for i in range(x.shape[0]):
            save_image(x[i], sample_path + f"sample_{i}.png")

    

    def load_model(self, file_name):
        self.net.load_state_dict(torch.load(file_name, weights_only=True))
    

def main():

    # ----------------------- Arguments -----------------------------
    dataset = sys.argv[1]           # dataset to train on
    num_epochs = int(sys.argv[2])   # number of epochs
    file_name = sys.argv[3]         # file name to save model
    save_every = int(sys.argv[4])   # save model every x epochs

    # if loading pre-trained model:
    load_params = False
    if len(sys.argv) > 5:
        load_params = True
        model_path = sys.argv[5]

    parent_dir = file_name + "/"

    # create directory for saving files
    if os.path.exists(parent_dir):
        raise ValueError("Directory already exists, change file name")
    
    os.mkdir(parent_dir)

    models_dir = parent_dir + "MODELS/"     # for saving model parameters
    os.mkdir(models_dir)
    samples_dir = parent_dir + "SAMPLES/"   # for saving samples
    os.mkdir(samples_dir)
    


    lr = 1e-4           # learning rate

    print(f"training net for: {num_epochs} epochs with lr: {lr}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"on device: {device}")

    score_network = Unet(dataset).to(device)

    ddpm = DDPM(beta_min=beta_min,
                beta_max=beta_max,
                T=T,
                net=score_network,
                device=device,
                dataset=dataset,
                batchsize=64)

    if load_params:
        ddpm.load_model(model_path)

    ddpm.train(num_epochs=num_epochs,
               lr=lr, log_interval=5,
               file_name=file_name,
               save_model_very=save_every,
               save_path=parent_dir,
               num_samples=64)

    print("training complete")

    samples = ddpm.sample(num_samples=64)
    samples = torch.permute(samples, (0,2,3,1)).numpy().squeeze()
    fig, axs = plt.subplots(8, 8, figsize=(16, 16))
    cmap = "binary_r" if dataset.lower() == "mnist" else None
    for i, ax in enumerate(axs.flat):
        ax.imshow(samples[i], cmap=cmap)
        ax.axis("off")
    
    plt.savefig(f"{file_name}_samples.png", dpi=100)




if  __name__ == "__main__":
    main()