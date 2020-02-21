# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 00:33:19 2020

@author: Abdullah
"""
#%%
import pandas as pd
#%%
dataset = pd.read_csv('Pre_AB186_Circuit_raw.csv')  # Change the Csv name here to your input file name
#%%
dataset2 = dataset[['Right_Shank_Ax','Right_Shank_Ay','Right_Shank_Az','Right_Shank_Gx','Right_Shank_Gy','Right_Shank_Gz','Right_Thigh_Ax','Right_Thigh_Ay','Right_Thigh_Az','Right_Thigh_Gx','Right_Thigh_Gy','Right_Thigh_Gz','Mode']].round(14)
#%%
dataset2.to_csv('AB186_Circuit_raw.csv', index = False)  # chane output name to be not the same as input to not over write it
#%%
dataset3 = pd.read_csv('AB186_Circuit_raw.csv') # Useless Line for Viualize not more