### feature extraction ###

##without_normal##

import numpy as np
import os
import librosa
from sklearn import preprocessing
from numpy import linalg as LA


train_list=os.listdir("./data/train/")
test_list=os.listdir("./data/test/")
val_list=os.listdir("./data/val/")

genre_label=["classical","jazz","pop","rock"]
# print train_list


#### Function Flux ####
def flux_difference (frames):

    flux=[]   
    stft_train=librosa.core.stft(y=frames)
    for i in range(stft_train.shape[1]):
        if i==0:
            flux=LA.norm(stft_train[:,i])
        else:
            flux=np.append(flux,LA.norm(stft_train[:,i]-stft_train[:,i-1]))
            
    flux=flux.reshape(1,stft_train.shape[1])
    return flux
        

average_win=40
ones=int(1200/average_win)

#### train ####
cent_train=[]
rolloff_train=[]
zerocross_train=[]
flux_train=[]
chroma_train=[]
train_label=[]

for idx,item  in enumerate(train_list):   
    print("idx is",idx)
    music_frames=np.load("./data/train/"+item)
    genre=item.split("_")[0]
    genre_idx=genre_label.index(genre)
    # print("genre_idx is",genre_idx)
    # print(type(ones))
    # print(ones)
    # print(train_label,"train_label")
    train_label=np.append(train_label,genre_idx*np.ones(ones),axis=0)

    if idx==0:
        cent_train=librosa.feature.spectral_centroid(y=music_frames)[:,:1200]
        rolloff_train=librosa.feature.spectral_rolloff(y=music_frames)[:,:1200]
        zerocross_train=2048*librosa.feature.zero_crossing_rate(y=music_frames)[:,:1200]
        flux_train=flux_difference(music_frames)[:,:1200]
        chroma_train=librosa.feature.chroma_stft(music_frames,norm=None)[:,:1200]
        
    else:
        cent_train=np.append(cent_train,librosa.feature.spectral_centroid(y=music_frames)[:,:1200],axis=1)
        rolloff_train=np.append(rolloff_train,librosa.feature.spectral_rolloff(y=music_frames)[:,:1200],axis=1)
        zerocross_train=np.append(zerocross_train,2048*librosa.feature.zero_crossing_rate(y=music_frames)[:,:1200],axis=1)
        flux_train=np.append(flux_train,flux_difference(music_frames)[:,:1200],axis=1)
        chroma_train=np.append(chroma_train,librosa.feature.chroma_stft(music_frames,norm=None)[:,:1200],axis=1)


###concatenate###

concat_feat=np.concatenate((cent_train,rolloff_train,zerocross_train,flux_train,chroma_train),axis=0)

average_steps=int(concat_feat.shape[1]/average_win) # 1200/40*320

feat_mean=[]
feat_std=[]
for step in range(average_steps):
    if step==0:

        feat_mean=np.mean(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1)
        print(feat_mean.shape)
        feat_std=np.std(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1)
    else:
        feat_mean=np.append(feat_mean,np.mean(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1),axis=1)
        feat_std=np.append(feat_std,np.std(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1),axis=1)



train_feat=np.concatenate((feat_mean,feat_std),axis=0)
print(" Train Feature extraction Complete....")        


###test###


cent_test=[]
rolloff_test=[]
zerocross_test=[]
flux_test=[]
chroma_test=[]
test_label=[]

for idx,item  in enumerate(test_list):   

    music_frames=np.load("./data/test/"+item)
    genre=item.split("_")[0]
    genre_idx=genre_label.index(genre)
    test_label=np.append(test_label,genre_idx*np.ones(ones),axis=0)

    if idx==0:
        cent_test=librosa.feature.spectral_centroid(y=music_frames)[:,:1200]
        rolloff_test=librosa.feature.spectral_rolloff(y=music_frames)[:,:1200]
        zerocross_test=2048*librosa.feature.zero_crossing_rate(y=music_frames)[:,:1200]
        flux_test=flux_difference(music_frames)[:,:1200]
        chroma_test=librosa.feature.chroma_stft(music_frames,norm=None)[:,:1200]
        
    else:
        cent_test=np.append(cent_test,librosa.feature.spectral_centroid(y=music_frames)[:,:1200],axis=1)
        rolloff_test=np.append(rolloff_test,librosa.feature.spectral_rolloff(y=music_frames)[:,:1200],axis=1)
        zerocross_test=np.append(zerocross_test,2048*librosa.feature.zero_crossing_rate(y=music_frames)[:,:1200],axis=1)
        flux_test=np.append(flux_test,flux_difference(music_frames)[:,:1200],axis=1)
        chroma_test=np.append(chroma_test,librosa.feature.chroma_stft(music_frames,norm=None)[:,:1200],axis=1)



concat_feat=np.concatenate((cent_test,rolloff_test,zerocross_test,flux_test,chroma_test),axis=0)


average_steps=int(concat_feat.shape[1]/average_win)
feat_mean=[]
feat_std=[]
for step in range(average_steps):
    if step==0:
        feat_mean=np.mean(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1)
        print(feat_mean.shape)
        feat_std=np.std(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1)
    else:
        feat_mean=np.append(feat_mean,np.mean(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1),axis=1)
        feat_std=np.append(feat_std,np.std(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1),axis=1)



test_feat=np.concatenate((feat_mean,feat_std),axis=0)
print("Test Feature extraction Complete....")        


####val####

cent_val=[]
rolloff_val=[]
zerocross_val=[]
flux_val=[]
chroma_val=[]
val_label=[]

for idx,item  in enumerate(val_list):   

    music_frames=np.load("./data/val/"+item)
    genre=item.split("_")[0]
    genre_idx=genre_label.index(genre)
    val_label=np.append(val_label,genre_idx*np.ones(ones),axis=0)

    if idx==0:
        cent_val=librosa.feature.spectral_centroid(y=music_frames)[:,:1200]
        rolloff_val=librosa.feature.spectral_rolloff(y=music_frames)[:,:1200]
        zerocross_val=2048*librosa.feature.zero_crossing_rate(y=music_frames)[:,:1200]
        flux_val=flux_difference(music_frames)[:,:1200]
        chroma_val=librosa.feature.chroma_stft(music_frames,norm=None)[:,:1200]
        
    else:
        cent_val=np.append(cent_val,librosa.feature.spectral_centroid(y=music_frames)[:,:1200],axis=1)
        rolloff_val=np.append(rolloff_val,librosa.feature.spectral_rolloff(y=music_frames)[:,:1200],axis=1)
        zerocross_val=np.append(zerocross_val,2048*librosa.feature.zero_crossing_rate(y=music_frames)[:,:1200],axis=1)
        flux_val=np.append(flux_val,flux_difference(music_frames)[:,:1200],axis=1)
        chroma_val=np.append(chroma_val,librosa.feature.chroma_stft(music_frames,norm=None)[:,:1200],axis=1)


concat_feat=np.concatenate((cent_val,rolloff_val,zerocross_val,flux_val,chroma_val),axis=0)
average_steps=int(concat_feat.shape[1]/average_win)

feat_mean=[]
feat_std=[]
for step in range(average_steps):
    if step==0:

        feat_mean=np.mean(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1)
        print(feat_mean.shape)
        feat_std=np.std(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1)
    else:
        feat_mean=np.append(feat_mean,np.mean(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1),axis=1)
        feat_std=np.append(feat_std,np.std(concat_feat[:,step*average_win:(step+1)*average_win],axis=1).reshape(concat_feat.shape[0],1),axis=1)



val_feat=np.concatenate((feat_mean,feat_std),axis=0)
print("Val Feature extraction Complete....")



###save###

np.save("./feature_data/train/train_sample.npy",train_feat.transpose())
np.save("./feature_data/train/train_label.npy",train_label)

np.save("./feature_data/test/test_sample.npy",test_feat.transpose())
np.save("./feature_data/test/test_label.npy",test_label)

np.save("./feature_data/val/val_sample.npy",val_feat.transpose())
np.save("./feature_data/val/val_label.npy",val_label)


print(" Feature extraction Complete....")


