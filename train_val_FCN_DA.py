# -*- coding: utf-8 -*-

import os
from keras.optimizers import Adadelta
from keras.callbacks import Callback
from keras import backend as K
import  numpy as np
import os.path
from math import ceil
import sys
from random import sample

sys.path.insert(0,'./cityscapesScripts/cityscapesscripts/evaluation')
from city_meanIU import city_meanIU

#parameters
image_size=[320,640]
source_batch_size=5
target_batch_size=5
batch_size=source_batch_size+target_batch_size
output_name='SYNTHIA_FCN_DA.h5'
class_number=22

#download pretrained SYNTHIA network
#you can start from scratch if you want of course
url='http://crcv.ucf.edu/data/adaptationseg/SYNTHIA_FCN.h5'
import urllib.request
import shutil
with urllib.request.urlopen(url) as response, open('./SYNTHIA_FCN.h5', 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

#set valid classes bool
index_array=np.zeros((class_number))
index_array[0:12]=1.
index_array[15]=1.
index_array[17]=1.
index_array[19]=1.
index_array[21]=1.

#image mean
image_mean = np.zeros((1,3,1,1), dtype=np.float32)
image_mean[:,0,:,:] = 103.939
image_mean[:,1,:,:] = 116.779
image_mean[:,2,:,:] = 123.68

#create network
from FCN_da import create_vgg16_FCN
seg_model=create_vgg16_FCN(image_size[0],image_size[1],class_number)
seg_model.load_weights('SYNTHIA_FCN.h5')
if os.path.isfile(output_name):
    seg_model.load_weights(output_name)


#SP-weighted classes-weighted pixelwise segmentation loss
def SP_pixelwise_loss(y_true,y_pred):
    y_true_label=y_true[:,:class_number,:,:]
    y_true_SP_weight=y_true[:,class_number:,:,:]
    
    y_pred=K.clip(y_pred,-50.,50.)#prevent overflow
    sample_num_per_class=K.sum(y_true_label,axis=[2,3],keepdims=True)
    class_ind=K.cast(K.greater(sample_num_per_class,0.),'float32')
    avg_sample_num_per_class=K.sum(sample_num_per_class,axis=1,keepdims=True)/K.sum(class_ind,axis=1,keepdims=True)
    sample_weight_per_class=avg_sample_num_per_class/(sample_num_per_class+0.1)
    exp_pred=K.exp(y_pred-K.max(y_pred,axis=1,keepdims=True))
    y_pred_softmax=exp_pred/K.sum(exp_pred,axis=1,keepdims=True)
    pixel_wise_loss=-K.log(y_pred_softmax)*y_true_label
    pixel_wise_loss=pixel_wise_loss*sample_weight_per_class
    weighter_pixel_wise_loss=K.sum(pixel_wise_loss,axis=1,keepdims=True)
    
    return K.mean(weighter_pixel_wise_loss*y_true_SP_weight)

#label distribution loss
def layout_loss_hard(y_true,y_pred):
    
    y_pred=K.clip(y_pred,-50.,50.)#prevent overflow
    exp_pred=K.exp(y_pred-K.max(y_pred,axis=1,keepdims=True))
    y_pred_softmax=exp_pred/K.sum(exp_pred,axis=1,keepdims=True)
    
    max_pred_softmax=K.max(y_pred_softmax,axis=1,keepdims=True)
    bin_pred_softmax_a=y_pred_softmax/max_pred_softmax
    bin_pred_softmax=bin_pred_softmax_a**6.
    
    final_pred=K.mean(bin_pred_softmax,axis=[2,3])
    final_pred=final_pred/(K.sum(final_pred,axis=1,keepdims=True)+K.epsilon())
    y_true_s=K.squeeze(y_true,axis=3)
    y_true_s=K.squeeze(y_true_s,axis=2)
    tier_wise_loss_v=-K.clip(K.log(final_pred),-500,500)*y_true_s
    return K.mean(K.sum(tier_wise_loss_v,axis=1))


#compile
seg_model.compile(optimizer=Adadelta(),
              loss={'output': SP_pixelwise_loss, 'output_2': layout_loss_hard},
              loss_weights={'output': 1.,'output_2':0.1})


def binarize_label(batch_seg):
    label_tensor_to_return=np.zeros((batch_seg.shape[0],class_number,image_size[0],image_size[1]),dtype=np.bool)
    count=0
    for i in range(batch_seg.shape[0]):
        label=np.squeeze(batch_seg[i,:,:])
        label_return=np.zeros((class_number,label.shape[0],label.shape[1]),dtype=np.bool)
        it = np.nditer(label, flags=['multi_index'])
        while not it.finished:
            if np.asscalar(it[0]) <= 12 or np.asscalar(it[0]) ==15 or np.asscalar(it[0]) ==17 or np.asscalar(it[0]) ==19 or np.asscalar(it[0]) ==21:
                label_return[it[0],it.multi_index[0],it.multi_index[1]]=True
            it.iternext()
        label_return = label_return[np.newaxis, ...]
        label_tensor_to_return[count,:,:,:]=label_return
        count+=1
    return label_tensor_to_return

def binarize_SP(batch_seg):
    max_dim=np.amax(batch_seg)
    label_tensor_to_return=np.zeros((batch_seg.shape[0],max_dim,image_size[0],image_size[1]),dtype=np.bool)
    count=0
    for i in range(batch_seg.shape[0]):
        label=np.squeeze(batch_seg[i,:,:])
        label_return=np.zeros((max_dim,label.shape[0],label.shape[1]),dtype=np.bool)
        it = np.nditer(label, flags=['multi_index'])
        while not it.finished:
            label_return[it[0]-1,it.multi_index[0],it.multi_index[1]]=True
            it.iternext()
        label_return = label_return[np.newaxis, ...]
        label_tensor_to_return[count,:,:,:]=label_return
        count+=1
    return label_tensor_to_return
        

print('Start loading files')
from warp_data import train_synthia_generator,val_synthia_generator,cityscape_im_generator
val_mean_IU_list=list()
(loaded_val_im,loaded_val_label)=val_synthia_generator[range(len(val_synthia_generator))]
loaded_val_im=loaded_val_im.astype('float32')-image_mean

#define training data generator
def myGenerator():
    rand_idx=np.random.permutation(len(train_synthia_generator))
    rand_idx_chunks = [rand_idx[x:x+source_batch_size] for x in range(0, len(rand_idx), source_batch_size)]
    while 1:
        for idx in rand_idx_chunks:
            (loaded_source_im,loaded_source_label)=train_synthia_generator[idx]
            loaded_source_im=loaded_source_im.astype('float32')-image_mean
            loaded_source_label=binarize_label(loaded_source_label).astype('float32')
            
            tar_idx=sample(range(len(cityscape_im_generator)),target_batch_size)
            loaded_target_im,loaded_SP_map,loaded_SP_annotation,loaded_target_obj_pre=cityscape_im_generator[tar_idx]
            
            loaded_SP_annotation=binarize_label(loaded_SP_annotation).astype('float32')
            loaded_SP_map=binarize_SP(loaded_SP_map)
            
            SP_pixelperSP_num=np.sum(loaded_SP_map.astype(np.float32),axis=(2,3),keepdims=True)
            avg_pixel_number=np.sum(SP_pixelperSP_num,axis=1,keepdims=True)/np.sum(SP_pixelperSP_num>0.,axis=1,keepdims=True)
            SP_weight=avg_pixel_number/SP_pixelperSP_num
            SP_weight[np.isinf(SP_weight)]=0.
            
            SP_weight_map=np.sum(SP_weight*loaded_SP_map,axis=1,keepdims=True)
            concat_target_GT=np.concatenate((loaded_SP_annotation,SP_weight_map),axis=1)
            
            #modify presence value
            loaded_target_obj_pre=loaded_target_obj_pre[:,:class_number]
            loaded_target_obj_pre=loaded_target_obj_pre/np.sum(loaded_target_obj_pre,axis=1,keepdims=True)
            loaded_target_obj_pre=np.pad(loaded_target_obj_pre,((len(idx),0),(0,0)),'constant',constant_values=0.)
            loaded_target_obj_pre = loaded_target_obj_pre[...,np.newaxis, np.newaxis]
    
            loaded_source_label=np.pad(loaded_source_label,((0,0),(0,1),(0,0),(0,0)),'constant',constant_values=1.)
            
            loaded_im=np.concatenate((loaded_source_im,loaded_target_im),axis=0)
            loaded_label=np.concatenate((loaded_source_label,concat_target_GT),axis=0)
            
            yield loaded_im,{'output':loaded_label,'output_2':loaded_target_obj_pre}
       
#validation callback
steps_per_epoch=int(ceil(float(len(train_synthia_generator))/source_batch_size))
class Validate_on_CityScape(Callback):
    def on_train_begin(self, logs={}):
        self.acc_iter = 0
        self.best_mean_IU = 0
        
    def on_batch_end(self, batch, logs={}):
        if np.isnan(logs.get('loss')): #Model contain NaN
            print('NaN detected, reloading model')
            self.model.compile(optimizer=Adadelta(),
              loss={'output': SP_pixelwise_loss, 'output_2': layout_loss_hard},
              loss_weights={'output': 1.,'output_2':0.1})
            self.model.load_weights(output_name)

    def on_batch_begin(self, batch, logs={}):
        
        if self.acc_iter%500==0:
            current_predicted_val=self.model.predict(loaded_val_im,batch_size=batch_size)
            current_predicted_val=current_predicted_val[0]
            predicted_val_class=np.argmax(current_predicted_val,axis=1)
            results_dict=city_meanIU(loaded_val_label,predicted_val_class)
            val_mean_IU=results_dict['averageScoreClasses']
            val_mean_IU_list.append(val_mean_IU)
            print('\nCurrently validation mean IU is '+str(val_mean_IU)+' while highest is '+str(self.best_mean_IU))
            if val_mean_IU>self.best_mean_IU:
                self.best_mean_IU=val_mean_IU
                self.model.save_weights(output_name,overwrite=True)

        self.acc_iter=self.acc_iter+1
        
print('Start training')
seg_model.fit_generator(myGenerator(),callbacks=[Validate_on_CityScape()], steps_per_epoch=steps_per_epoch, epochs=60)
    
