from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
class DE(nn.Module):
    def __init__(self,device="cuda",
                 dropouts=0.1,input =3000,output1 =1500,output2=500):
        super(DE, self).__init__()
        self.device=device
        self.ru = nn.LeakyReLU()  #
        self.drop = nn.Dropout(p=dropouts)

        self.enc1 = nn.Linear(input, output1)
        self.enc2 = nn.Linear(output1, output2)
        self.dec1 = nn.Linear(output2, output1)
        self.dec2 = nn.Linear(output1, input)
        self.MSELoss = nn.MSELoss()
        self.norm1 = nn.LayerNorm(output1)
        self.norm2 = nn.LayerNorm(output2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.eye_(m.weight)
                m.bias.data.zero_()

    def netclean(self, x):
        enc1 = self.enc1(x)
        enc1 = self.ru(enc1)
        #enc1 =self.norm1(enc1)

        z = self.enc2(enc1)
        z = self.ru(z)
        #z = self.norm2(z)

        dec1 = self.dec1(z)
        dec1 = self.ru(dec1)
        #dec1 = self.norm1(dec1)

        dec2 = self.dec2(dec1)
        #dec2 = self.ru(dec2)
        return z, enc1,dec1,dec2

    def netnoise(self, x):
        x = x + (torch.rand(x.shape) * 1).to(device=self.device)

        #enc1 =self.drop(x)
        enc1 = self.enc1(x)
        enc1 = self.ru(enc1)
        #enc1 = self.norm1(enc1)

        #z = self.drop(enc1)
        z = self.enc2(enc1)
        z = self.ru(z)
        #z = self.norm2(z)

        #dec1 = self.drop(z)
        dec1 = self.dec1(z)
        dec1 = self.ru(dec1)
        #dec1 = self.norm1(dec1)


        #dec2 = self.drop(dec1)
        dec2 = self.dec2(dec1)
        #dec2 = self.ru(dec2)
        return z, enc1,dec1,dec2

    def cosine(self, x):
        #cosine =np.corrcoef(x.data.cpu().numpy())
        cosine = torch.pow(torch.sum(x ** 2.0, dim=1), 0.5)
        cosine = (x.t() / cosine).t()
        cosine = torch.mm(cosine, cosine.t())
        return cosine

    def _build_lossDE(self, x, znoise, enc1, dec1,dec2, u):
        loss1 = self.MSELoss(dec2, x)
        loss1 += self.MSELoss(enc1, dec1)
        true_cosine = self.cosine(znoise)
        noise_cosine = self.cosine(u.to(self.device))
        loss3 = self.MSELoss(noise_cosine, true_cosine)
        return loss1+loss3,loss1


class VAE(nn.Module):
    def __init__(self,device="cuda",
                 dropouts=0.1,input =3000,output1 =1500,output2=500):
        super(VAE, self).__init__()
        self.device=device
        self.ru = nn.LeakyReLU()  #
        self.drop = nn.Dropout(p=dropouts)

        self.enc1 = nn.Linear(input, output1)
        self.enc2 = nn.Linear(output1, output2)
        self.mu = nn.Linear(output1, output2)
        self.logvar = nn.Linear(output1, output2)
        self.dec1 = nn.Linear(output2, output1)
        self.dec2 = nn.Linear(output1, input)
        self.MSELoss = nn.MSELoss()
        self.KLDivLoss = nn.KLDivLoss()
        self.norm1 = nn.LayerNorm(output1)
        self.norm2 = nn.LayerNorm(output2)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.eye_(m.weight)
                m.bias.data.zero_()

    def netclean(self, x):
        enc1 = self.enc1(x)
        enc1 = self.ru(enc1)
        # enc1 =self.norm1(enc1)

        z = self.enc2(enc1)
        z = self.ru(z)
        # z = self.norm2(z)

        dec1 = self.dec1(z)
        dec1 = self.ru(dec1)
        # dec1 = self.norm1(dec1)

        dec2 = self.dec2(dec1)
        # dec2 = self.ru(dec2)
        return z, enc1, dec1, dec2

    def netnoise(self, x):
        x = x + (torch.rand(x.shape) * 1).to(device=self.device)

        #enc1 =self.drop(x)
        enc1 = self.enc1(x)
        enc1 = self.ru(enc1)
        #enc1 = self.norm1(enc1)

        z_mean = self.mu(enc1)
        z_log_var = self.logvar(enc1)
        std = torch.exp(0.5 * z_log_var)
        u = torch.randn_like(std)
        z = u * std + z_mean

        dec1 = self.dec1(z)
        dec1 = self.ru(dec1)
        #dec1 = self.norm1(dec1)


        #dec2 = self.drop(dec1)
        dec2 = self.dec2(dec1)
        dec2 = self.ru(dec2)
        sd = torch.tensor([0.1]).to(self.device)
        dist = torch.distributions.Normal(dec2, sd)
        sample = dist.rsample()
        mean = self.sigmoid(sample)
        return z, dec2,sd,mean, z_mean, z_log_var

    def cosine(self, x):
        cosine = torch.pow(torch.sum(x ** 2.0, dim=1), 0.5)
        cosine = (x.t() / cosine).t()
        cosine = torch.mm(cosine, cosine.t())
        return cosine

    def _build_lossVAE(self, x, zclean, enc1, sd_d,mean_d, u, mu, logvar):
        dist2 = torch.distributions.Normal(mean_d, sd_d)
        loss1 = dist2.log_prob(x)

        #loss1 += self.MSELoss(enc1, dec1)
        # compute KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = -1 * torch.sum(loss1) + KLD
        true_cosine = self.cosine(zclean)
        noise_cosine = self.cosine(u.to(self.device))
        loss3 = self.MSELoss(noise_cosine, true_cosine)
        return  loss, loss+loss3












