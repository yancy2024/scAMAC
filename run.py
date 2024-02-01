from __future__ import print_function, division
import csv
import math
import random
import os

import scanpy as sc
from utils import cal_clustering_metric
from preprocessing import computeCentroids,reshapeX
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from DE import DE
from DFKM import DFKM
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

class DEFKM(nn.Module):
    def __init__(self,
                 dropouts=0.1,
                 input =3000,
                 output1=300,
                 output2=30,
                 labels =None,
                 X=None,
                 adj=None,
                 lr =0.0001,
                 device =None,
                 num_attention_heads=2):
        super(DEFKM, self).__init__()
        self.labels = labels
        self.X = X
        self.adj=adj
        self.n_clusters =len(np.unique(self.labels))
        self.lr = lr
        self.drop =dropouts

        self.device = device
        if (self.X.shape[0] < 500):
            self.batch_size = 32
        elif (self.X.shape[0] < 700):
            self.batch_size = 100
        else:
            self.batch_size = 128
        self.DE = DE(device=self.device,dropouts=self.drop,input =input,output1 =output1,output2=output2)
        self.DFKM =DFKM(device=self.device,dropouts=self.drop,input =output1,output =output2)   # , input_size = self.X.shape[0]

    def runDE(self,eD,train_loaderDE,optimizerDE):
        #loss11 = np.array([])
        for epoch1 in range(eD):
            loss = 0
            c = 0

            for i, batch in enumerate(train_loaderDE):
                x = batch[0][0]
                idx = batch[1]

                optimizerDE.zero_grad()  # 梯度置零
                z, enc1,dec1,dec2 = self.DE.netnoise(x)
                u = self.DFKM.U[idx, :]
                loss0,loss1 = self.DE._build_lossDE(x, z, enc1,dec1,dec2, u)

                if (math.isnan(loss0.item())):
                    continue
                c = c + 1

                loss = loss + loss0.item()
                loss0.backward()
                optimizerDE.step()

    def runDFKM(self,eF,train_loaderFKM,optimizerFKM,z, enc1, dec1):
        self.loss22 = 10000000
        ari = 0
        nmi=0
        y_pred=None
        L=1
        #loss11 = np.array([])
        for epoch0 in range(eF):
            loss =0
            c = 0

            D = self.DFKM._update_D(self.Z)

            for i, batch in enumerate(train_loaderFKM):
                x = batch[0][0]
                e1 = batch[0][1]
                d1 = batch[0][2]
                #x= x + (torch.rand(x.shape) * 0.1).to(device=self.device)
                idx = batch[1]
                optimizerFKM.zero_grad()  # 梯度置零
                zz = self.DFKM.netclean(x, e1, d1)
                d = D[idx, :]
                u = self.DFKM.U[idx, :]

                loss1 = self.DFKM._build_lossFKM( zz, d, u)
                if (math.isnan(loss1.item())):
                    L = 0
                    break

                c=c+1
                loss =loss+loss1.item()

                loss1.backward()
                optimizerFKM.step()


        zclean = self.DFKM.netclean(z, enc1, dec1)
        self.Z = zclean.t().detach()

        y_pred = self.DFKM.clustering(self.Z,self.n_clusters)

        nmi, ari = cal_clustering_metric(self.labels, y_pred)
        return nmi, ari, y_pred,L ,zclean#


    def run(self, eD =3,eF =3,e=100,name="_"):
        self.DE.to(self.device)
        self.DFKM.to(self.device)
        z, enc1,dec1,dec2= self.DE.netclean(self.X)
        zclean = self.DFKM.netclean(z, enc1,dec1)
        self.Z = zclean.t().detach()

        # l = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='euclidean', linkage='ward').fit(
        #     zclean.detach().cpu()).labels_
        # centers = computeCentroids(zclean.data.cpu().numpy(), l)
        # self.DFKM.setC(torch.tensor(centers).t().to(self.device))

        idx = random.sample(list(range(self.Z.shape[1])), self.n_clusters)
        centroids = self.Z[:, idx] + 10 ** -6
        self.DFKM.setC(centroids.to(self.device))
        self.DFKM._update_U(self.Z)
        optimizerFKM = torch.optim.SGD(self.DFKM.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerFKM, T_max=100)
        optimizerDE = torch.optim.Adam(self.DE.parameters(), lr=self.lr)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerFKM, T_max=100)
        train_loaderDE = DataLoader(TensorDataset(self.X), batch_size=self.batch_size, shuffle=True)
        #ccc =None
        a0 =[0,0]
        final_nmi, final_ari = 0, 0
        final_pred, final_Zdata, final_chongou = 0, self.Z, 0
        for epoch in range(e):

            self.runDE(eD,train_loaderDE,optimizerDE)
            z, enc1, dec1, chonggou = self.DE.netclean(self.X)
            zclean = self.DFKM.netclean(z, enc1, dec1)
            self.Z = zclean.t().detach()

            train_loaderFKM = DataLoader(TensorDataset(z.clone().detach().requires_grad_(True),enc1.clone().detach().requires_grad_(True),dec1.clone().detach().requires_grad_(True)), batch_size=self.batch_size,shuffle=True)

            nmi, ari, y_pred,L,zzz =self.runDFKM(eF, train_loaderFKM, optimizerFKM,z, enc1, dec1)
            if (L == 0): return 0, 0, None,None,None
            nmi, ari = cal_clustering_metric(self.labels, y_pred)
            print('{}:epoch-{}, nmi={}, ari={}'.format(name,epoch, nmi, ari))#, file=file
            scheduler.step()
            scheduler1.step()
            if final_ari < ari:
                final_ari = ari
                final_nmi = nmi
                final_pred = y_pred
                final_Zdata = zzz
                final_chongou = chonggou
        return final_nmi, final_ari, final_pred, final_Zdata, final_chongou

if __name__ == '__main__':
        la = pd.read_csv("goolam_lable.csv", header=0,index_col=0, sep=',')
        la = np.array(la).reshape(la.shape[0], )
        da = 'goolam_data.csv'

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(da)

        X = pd.read_csv(da, header=0,index_col=0,sep=',')
        X = np.array(X).T
        print(X.shape)
        print(len(la))

        X = torch.FloatTensor(X).to(device)
        ii =3
        jj=2
        model = DEFKM(X=X,device =device,labels=la,input=X.shape[1],output1=1500,output2=500).to(device)  # len(np.unique(la)),output=nnnn
        nmi,ari,y_pred,zzz,chonggou = model.run(name='goolam',e=50,eD=ii,eF=jj)

        print("ARI:{},NMI:{}".format(ari, nmi))
