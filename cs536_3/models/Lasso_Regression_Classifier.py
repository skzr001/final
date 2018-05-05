# Lasso_Regression_Classifier
import numpy as np
from sklearn import linear_model
from numpy import linalg as LA
from collections import Counter

class Lasso_Regression_Classifier(object):

  def __init__(self,D,L,train_label,average_win):
      self.D = D # train_samples as dictionary (feat x samples)
      self.L = L # regularization parameter
      self.num_genre=4
      self.train_label_0=np.where(train_label==0, 1, 0)
      self.train_label_1=np.where(train_label==1, 1, 0)
      self.train_label_2=np.where(train_label==2, 1, 0)
      self.train_label_3=np.where(train_label==3, 1, 0)
      self.samples_per_music=int(1200/average_win)

      
  

  def sparse_code_w(self,i):
      ####STUDENT CODE####
      # print("lasso step starts!")
      from sklearn import linear_model
      clf = linear_model.Lasso(alpha=self.L, normalize = True)
      clf.fit(self.D,self.test_samples[i])
      w = clf.coef_ 
      return w

  def classifier(self,w,i):
      
      ####STUDENT CODE####
      residual = []
      residual.append(np.linalg.norm(self.test_samples[i] - np.matmul(self.D,np.multiply(w,self.train_label_0))))
      residual.append(np.linalg.norm(self.test_samples[i] - np.matmul(self.D,np.multiply(w,self.train_label_1))))
      residual.append(np.linalg.norm(self.test_samples[i] - np.matmul(self.D,np.multiply(w,self.train_label_2))))
      residual.append(np.linalg.norm(self.test_samples[i] - np.matmul(self.D,np.multiply(w,self.train_label_3))))
      genre_estimate = residual.index(min(residual))
      #print(residual)
      #print("genre_estimate is", genre_estimate)
      return genre_estimate
  
  def most_common(self,lst):
    data=Counter(lst)
    return data.most_common(1)[0][0]



  def predict(self,test_samples):
      ##sparse code learning##
      ##classifier##
      self.test_samples=test_samples  
    
      sample_prediction=[]
      for i in range(test_samples.shape[0]): # number of samples
          if(i%100 == 0):
                print(i," th sample")
          w=self.sparse_code_w(i)
          # print(w)
          genre_estimate=self.classifier(w,i)
          sample_prediction=np.append(sample_prediction,genre_estimate)
   
      ###majority voting###
      
      steps=int(len(sample_prediction)/self.samples_per_music)
      #print(steps)
      for step in range(steps):
          majority_label=self.most_common(sample_prediction[step*self.samples_per_music:step*self.samples_per_music+self.samples_per_music])
          sample_prediction[step*self.samples_per_music:step*self.samples_per_music+self.samples_per_music]=majority_label*np.ones(self.samples_per_music)
       
      
      return sample_prediction


    


    
  
