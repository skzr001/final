import numpy as np
from sklearn import linear_model

from collections import Counter
def most_common(lst):
    data=Counter(lst)
    return data.most_common(1)[0][0]

train_path="./feature_data/train/"
test_path="./feature_data/test/"

train_sample=np.load(train_path+"train_sample.npy")
train_label=np.load(train_path+"train_label.npy")

test_sample=np.load(test_path+"test_sample.npy")
test_label=np.load(test_path+"test_label.npy")


logreg=linear_model.LogisticRegression()
logreg.fit(train_sample,train_label)
prediction=logreg.predict(test_sample)
####majority vote####
sample_size=30 # per....music.
step=int(len(prediction)/sample_size)

for idx in range(step):
    majority_label=most_common(prediction[idx*sample_size:idx*sample_size+sample_size])
    prediction[idx*sample_size:idx*sample_size+sample_size]=majority_label*np.ones(sample_size)#replace

test_num=test_sample.shape[0] #number of test sample
test_acc = np.sum(test_label == prediction)/(1.0*test_num)
print(test_acc)
