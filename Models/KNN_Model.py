# K-Nearest Neighbors (K-NN)
 
#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set()

#%% Importing the dataset
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('AB186_Circuit_002_raw.csv')

X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values

dataset_2 = pd.read_csv('AB186_Circuit_001_raw.csv')
X_test = dataset_2.iloc[:, :-1].values
y_test = dataset_2.iloc[:, -1].values
_, X_test, _, y_test = train_test_split(X_test, y_test, test_size = 0.4)

# #%% Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#%% Predicting the Test set results
y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

#%% Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)
#accuracy_train = classifier.score(X_train, y_train)
#accuracy_set = classifier.score(X_test, y_test)
#print("trainset_Accuracy is : ",accuracy_train*100,'%')
#print("testset_Accuracy is : ",accuracy_set*100,'%')
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 
Accuracy_train=format(accuracy(cm_train)*100,'.3f')
Accuracy_test=format(accuracy(cm_test)*100,'.3f')
print("Accuracy of train_set is : "+ Accuracy_train +' %')
print("Accuracy of test_set is : "+ Accuracy_test +' %')

#%% Visualising the Training set results #2-4 3-4 4-3
"""for i in [cnt for cnt in range(10) if cnt != 1]:
  plt.title(str(i))
  scatter=plt.scatter(X_train[:,1],X_train[:,i],c=y_pred_train,cmap='rainbow',s=2)
  plt.legend(*scatter.legend_elements())
  plt.xlabel('X1')
  plt.ylabel('X2')
  plt.show( block=False )"""

plt.title('Kernel SVM (Training set) Accuracy is ')#+ Accuracy_train +' %')
scatter=plt.scatter(X_train[:,4],X_train[:,3],s=.5)#,c=y_pred_train,cmap='rainbow',s=2)
plt.legend(*scatter.legend_elements(),bbox_to_anchor=(1, 0.45))
plt.xlabel('X1')
plt.ylabel('X2')

#%% Visualising the Testing set results
plt.title('Kernel SVM (Testing set) Accuracy is '+ Accuracy_test +' %')
scatter=plt.scatter(X_test[:,4],X_test[:,3],c=y_pred_test,cmap='rainbow',s=2)
plt.legend(*scatter.legend_elements(),bbox_to_anchor=(1, 0.45))
plt.xlabel('X1')
plt.ylabel('X2')

#%% save the model to disk
pickle.dump(classifier, open('C:/Users/Abdullah/Desktop/SVM_Model/model.pkl', 'wb'))
#%%
"""#%% Visualising the Training set results
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(48)]).T
# Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
pred = classifier.predict(Xpred.reshape(-1, 1)).reshape(X1.shape)   # is a matrix of 0's and 1's !
plt.contourf(X1, X2, pred, alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, s=.5)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#%% Visualising the Test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('tomato', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, s=.5)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""