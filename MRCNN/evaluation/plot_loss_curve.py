#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:27:17 2024

@author: moreau
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

metrics_df = pd.read_json("/path/to/model/output/metrics.json", orient="records", lines=True)
mdf = metrics_df.sort_values("iteration")


mdf = metrics_df.sort_values("iteration")
mdf1 = mdf[~mdf["total_loss"].isna()]
iter_train = mdf1["iteration"]
loss_train = mdf1["total_loss"]
if "validation_loss" in mdf.columns:
    mdf2 = mdf[~mdf["validation_loss"].isna()]
    iter_val = mdf2["iteration"]
    loss_val = mdf2["validation_loss"]
    
iter_val = iter_val.tolist()
iter_train = iter_train.tolist()
loss_val= loss_val.tolist()
loss_train = loss_train.tolist()
    

epoque = 1
loss_epoque= []
val_loss = []
for e in range(len(iter_val)):
    if iter_val[e]<epoque*375:
        loss_epoque.append(loss_val[e])
    else:
        val_loss.append(np.sum(loss_epoque)/len(loss_epoque))
        loss_epoque = []
        epoque +=1
        
epoque = 1
loss_epoque= []
train_loss = []
for e in range(len(iter_train)):
    if iter_train[e]<epoque*375:
        loss_epoque.append(loss_train[e])
    else:
        train_loss.append(np.sum(loss_epoque)/len(loss_epoque))
        loss_epoque = []
        epoque +=1
    
epoques = range(len(train_loss))
    
plt.rcParams["figure.figsize"] = (16,12)

plt.plot(epoques, train_loss, c="firebrick", linewidth=3)

plt.plot(epoques, val_loss, c="navy", linewidth=3, linestyle='--')
plt.legend(['entrainement','validation'], fontsize=30)
plt.xlabel('Epoques', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


