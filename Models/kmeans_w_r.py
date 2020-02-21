# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
!pip install -U -q PyDrive
import seaborn as sns
sns.set()

#from google.colab import files
#uploaded = files.upload()
from google.colab import drive
drive.mount('/content/gdrive')

imu_data = pd.read_csv('/content/gdrive/My Drive/Datasets/walking_running.csv')
print(imu_data)

plt.scatter(imu_data['Gyroscope_X'],imu_data['Gyroscope_Y'],s=1)
plt.xlabel('Gyroscope_X')
plt.ylabel('Gyroscope_Y')

def create_model():
  x = preprocessing.scale(imu_data)
  kmeans = KMeans(2)
  kmeans.fit(x)
  result =kmeans.fit_predict(x)
  return result
pred=create_model()
print(pred)

scatter=plt.scatter(imu_data['Gyroscope_X'],imu_data['Gyroscope_Y'],c=pred,cmap='rainbow',s=2)
plt.legend(*scatter.legend_elements())
plt.xlabel('Gyroscope_X')
plt.ylabel('Gyroscope_Y')

imu_data['Class'] = pred
imu_data.to_csv('output.csv')
cols = list(pd.read_csv("output.csv", nrows =1))
a= pd.read_csv("output.csv", usecols =[i for i in cols if i != 'Unnamed: 0'])
print(a)

"""# New Section"""

!cp output.csv "/content/gdrive/My Drive/Outputs"

!pip install h5py pyyaml

#saving models
model = create_model()

model.fit()

# Save entire model to a HDF5 file
model.save('my_model.h5')