# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-


import  numpy as np
from warp_data import test_cityscape_generator
import sys

sys.path.insert(0,'./cityscapesScripts-master/cityscapesscripts/evaluation')
from city_meanIU import city_meanIU

image_size=[320,640]
batch_size=5
image_mean = np.zeros((1,3,1,1), dtype=np.float32)
image_mean[:,0,:,:] = 103.939
image_mean[:,1,:,:] = 116.779
image_mean[:,2,:,:] = 123.68
class_number=22


print('Start loading image')
loaded_test_im=test_cityscape_generator[range(len(test_cityscape_generator))]
loaded_test_im,loaded_test_label=loaded_test_im.astype('float32')-image_mean

#create network
from FCN_da import create_vgg16_FCN
seg_model=create_vgg16_FCN(image_size[0],image_size[1],class_number)

#test pretrained baseline model mean IoU
seg_model.load_weights('SYNTHIA_FCN.h5')
current_predicted_test=seg_model.predict(loaded_test_im,batch_size=batch_size)
current_predicted_test=current_predicted_test[0]
predicted_test_class=np.argmax(current_predicted_test,axis=1)
results_dict=city_meanIU(loaded_test_label,predicted_test_class)
test_mean_IU=results_dict['averageScoreClasses']
print('Baseline test mean IoU is '+str(test_mean_IU))


#test trained AdaptationSeg mean IoU
seg_model.load_weights('SYNTHIA_FCN_DA.h5')
current_predicted_test=seg_model.predict(loaded_test_im,batch_size=batch_size)
current_predicted_test=current_predicted_test[0]
predicted_test_class=np.argmax(current_predicted_test,axis=1)
results_dict=city_meanIU(loaded_test_label,predicted_test_class)
test_mean_IU=results_dict['averageScoreClasses']
print('AdaptationSeg test mean IoU is '+str(test_mean_IU))
