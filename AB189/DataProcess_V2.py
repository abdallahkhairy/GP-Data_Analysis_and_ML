"""
@author: Mazen Refaat Ali Metwaly
@email:mazen.refaat23@gmail.com

TITLE: -----> Cleaning raw data <----- 
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import glob
import statistics as st



def ProcessData(sample_data):
    
    values = list()
    values.append(list(sample_data.Right_Shank_Ax))
    values.append(list(sample_data.Right_Shank_Ay))
    values.append(list(sample_data.Right_Shank_Az))
    values.append(list(sample_data.Right_Shank_Gx))
    values.append(list(sample_data.Right_Shank_Gy))
    values.append(list(sample_data.Right_Shank_Gz))
    values.append(list(sample_data.Right_Thigh_Ax))
    values.append(list(sample_data.Right_Thigh_Ay))
    values.append(list(sample_data.Right_Thigh_Az))
    values.append(list(sample_data.Right_Thigh_Gx))
    values.append(list(sample_data.Right_Thigh_Gy))
    values.append(list(sample_data.Right_Thigh_Gz))
    
    state = list(sample_data.Mode)
    
    TEMP_min = list()
    TEMP_max = list()
    TEMP_mean = list()
    TEMP_state = list()
    TEMP_dev = list()
    
    for m in range(12):
    
        #for Right_Shank_Ax
        
    
        temp_min = list()
        temp_max = list()
        temp_mean = list()
        temp_state = list()
        temp_dev = list()
        i = 0
        while i < len(values[m]):
            j = i + 240
            
            if j <= len(values[m]):
                pass        
            else:
                j = len(values[m])
             
                
            mi = np.min(values[m][i:j])
            ma = np.max(values[m][i:j])
            mean = st.mean(values[m][i:j])
            dev = st.stdev(values[m][i:j])
            try:
                x = st.mode(state[i:j])
            except:
                x = state[i]
            
            temp_mean.append(mean)
            temp_min.append(mi)
            temp_max.append(ma)
            temp_dev.append(dev)
            temp_state.append(x)
            
            i = i + 50
            #i = i + 80
            
        TEMP_mean.append(temp_mean)
        TEMP_min.append(temp_min)
        TEMP_max.append(temp_max)
        TEMP_dev.append(temp_dev)
        TEMP_state.append(temp_state)
        
        
        
    # intialise data of lists.
    data = {'mean_R_S_Ax': TEMP_mean[0],
            'min_R_S_Ax': TEMP_min[0],
            'max_R_S_Ax': TEMP_max[0],
            'dev_R_S_Ax': TEMP_dev[0],
            'mean_R_S_Ay': TEMP_mean[1],
            'min_R_S_Ay': TEMP_min[1],
            'max_R_S_Ay': TEMP_max[1],
            'dev_R_S_Ay': TEMP_dev[1],
            'mean_R_S_Az': TEMP_mean[2],
            'min_R_S_Az': TEMP_min[2],
            'max_R_S_Az': TEMP_max[2],
            'dev_R_S_Az': TEMP_dev[2],
            'mean_R_S_Gx': TEMP_mean[3],
            'min_R_S_Gx': TEMP_min[3],
            'max_R_S_Gx': TEMP_max[3],
            'dev_R_S_Gx': TEMP_dev[3],
            'mean_R_S_Gy': TEMP_mean[4],
            'min_R_S_Gy': TEMP_min[4],
            'max_R_S_Gy': TEMP_max[4],
            'dev_R_S_Gy': TEMP_dev[4],
            'mean_R_S_Gz': TEMP_mean[5],
            'min_R_S_Gz': TEMP_min[5],
            'max_R_S_Gz': TEMP_max[5],
            'dev_R_S_Gz': TEMP_dev[5],
            'mean_R_T_Ax': TEMP_mean[6],
            'min_R_T_Ax': TEMP_min[6],
            'max_R_T_Ax': TEMP_max[6],
            'dev_R_T_Ax': TEMP_dev[6],
            'mean_R_T_Ay': TEMP_mean[7],
            'min_R_T_Ay': TEMP_min[7],
            'max_R_T_Ay': TEMP_max[7],
            'dev_R_T_Ay': TEMP_dev[7],
            'mean_R_T_Az': TEMP_mean[8],
            'min_R_T_Az': TEMP_min[8],
            'max_R_T_Az': TEMP_max[8],
            'dev_R_T_Az': TEMP_dev[8],
            'mean_R_T_Gx': TEMP_mean[9],
            'min_R_T_Gx': TEMP_min[9],
            'max_R_T_Gx': TEMP_max[9],
            'dev_R_T_Gx': TEMP_dev[9],
            'mean_R_T_Gy': TEMP_mean[10],
            'min_R_T_Gy': TEMP_min[10],
            'max_R_T_Gy': TEMP_max[10],
            'dev_R_T_Gy': TEMP_dev[10],
            'mean_R_T_Gz': TEMP_mean[11],
            'min_R_T_Gz': TEMP_min[11],
            'max_R_T_Gz': TEMP_max[11],
            'dev_R_T_Gz': TEMP_dev[11],
            'state': TEMP_state[11]}
     
    
    
    # Create DataFrame
    new = pd.DataFrame(data)  
    
    
    #new.to_csv (r'C:\Users\user\Desktop\AB189\DataSet.csv', index = False, header=True)
    return new




def func():
    path = r'C:\Users\user\Desktop\AB189\Raw'
    all_files = glob.glob(path + "/*.csv")
    list_files = list()
    
    i = 1

    for fileName in all_files:
        df = pd.read_csv(fileName, header = 0)
        df2 = df[['Right_Shank_Ax', 'Right_Shank_Ay', 'Right_Shank_Az', 'Right_Shank_Gx', 'Right_Shank_Gy', 'Right_Shank_Gz', 'Right_Thigh_Ax', 'Right_Thigh_Ay', 'Right_Thigh_Az', 'Right_Thigh_Gx', 'Right_Thigh_Gy', 'Right_Thigh_Gz',  'Mode']].round(14)
        
        x = ProcessData(df2)
        print("file Number: ",i ," --- size:" , x.shape)
        i = i + 1
        
        list_files.append(x)
        
        
    dataset = pd.concat(list_files, axis = 0, ignore_index = True)

    

    print(dataset)
    dataset.to_csv (r'C:\Users\user\Desktop\AB189\DataSet_AB189.csv',index = False, header=True)
   
    
    
#--------------------> MAIN <----------------------------
def main():
    func()
  
    
    
    
if __name__== "__main__":
  main()
    
    
    
    
    
    
