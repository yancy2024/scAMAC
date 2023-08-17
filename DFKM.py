from __future__ import print_function, division

from sklearn.cluster import SpectralClustering

from preprocessing import computeCentroids
from  utils import distance
import torch
import torch.nn as nn
#from SelfAttention import  SelfAttention
from MSAMLP import Block as CBAM

class DFKM(nn.Module):
    def __init__(self,device="cuda",dropouts=0.1,input =1500,output =500):
        super(DFKM, self).__init__()
        self.gamma = 1
        self.sigma = 0.01
        self.device =device
        self.ru = nn.LeakyReLU()  #
        self.drop = nn.Dropout(p=dropouts)
        self.enc1 = nn.Linear(input, output)
        self.enc2 = nn.Linear(input, output)

        self.atten3 = CBAM(3,output)
        # self.SelfAttention = SelfAttention(num_attention_heads=1, input_size=output,
        #                                    hidden_size=output)


        #self.SelfAttention = SelfAttention(3)
        self.MSELoss = nn.MSELoss()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
#                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.eye_(m.weight)
                if m != 0:
                    m.bias.data.zero_()
    def setC(self,c):
        self.centroids=c

    def netnoise(self, z, e1, d1):

        e1 = self.enc1(e1)
        #e1 = self.ru(e1)
        d1 = self.enc2(d1)
        #d1 = self.ru(d1)
        z = self.atten3(torch.stack([z, e1, d1]))
        return z


    def netclean(self, z,e1,d1):

        e1 = self.enc1(e1)
        #e1 = self.ru(e1)

        #z = self.enc3(z)
        d1 = self.enc2(d1)
        #d1 = self.ru(d1)
        z = self.atten3(torch.stack([z, e1, d1]))
        return z

    def cosine(self, x):
        cosine = torch.pow(torch.sum(x ** 2.0, dim=1), 0.5)
        cosine = (x.t() / cosine).t()
        cosine = torch.mm(cosine, cosine.t())
        return cosine

    def _build_lossFKM(self, zz, d, u, lam = 0.1):
        size = zz.shape[0]
        t = d * u
        distances = distance(zz.t(), self.centroids)
        loss1 =  torch.trace(distances.t().matmul(t)) / (size)  #.matmul(t)乘法，torch.trace求对角线之和
        return loss1

    def _update_D(self, Z):
        if self.sigma is None:
            return torch.ones([Z.shape[1], self.centroids.shape[1]]).to(self.device)
        else:
            distances = distance(Z, self.centroids, False)
            return (1 + self.sigma) * (distances + 2 * self.sigma) / (2 * (distances + self.sigma))

    def clustering(self, Z,n_clusters):
        D = self._update_D(Z)

        # l = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", assign_labels="discretize",
        #                        random_state=0).fit_predict(adj)
        # centers = computeCentroids(Z.t().data.cpu().numpy(), l)
        # self.setC(torch.tensor(centers).t().to(self.device))

        T = D *self.U
        self.centroids = Z.matmul(T) / T.sum(dim=0).reshape([1, -1])

        self._update_U(Z)
        _, y_pred = self.U.max(dim=1)
        y_pred = y_pred.detach().cpu() + 1
        y_pred = y_pred.numpy()

        return y_pred

    def _update_U(self, Z):
        if self.sigma is None:
            distances = distance(Z, self.centroids, False)
        else:
            distances = self.adaptive_loss(distance(Z, self.centroids, False), self.sigma)
        U = torch.exp(-distances / self.gamma)
        self.U = U / U.sum(dim=1).reshape([-1, 1])

    def adaptive_loss(self,D, sigma):
        return (1 + sigma) * D * D / (D + sigma)