# Load libraries 
import os.path

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# load dataset or download and save locally
if os.path.isfile('iris.csv'):
  dataset = read_csv('iris.csv',index_col=0)
  print ("File loaded locally")
else:
  url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
  names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
  dataset = read_csv(url, names=names)
  dataset.to_csv('iris.csv')
  print("File downloaded and stored in iris.csv")

print(dataset)