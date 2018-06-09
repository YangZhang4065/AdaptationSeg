# -*- coding: utf-8 -*-
import  numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy.io import loadmat
from glob import glob

image_size=[640,320]

def load_image_single(f_path):
    im = Image.open(f_path).convert('RGB')
    width_side = im.size[0]
    new_h=width_side/2
    im = im.crop(
        (
            0,
            im.size[1]/2-new_h/2,
            width_side,
            im.size[1]/2+new_h/2
        )
    )
    im = im.resize( (image_size[0],image_size[1]),Image.LANCZOS)
    in_ = np.array(im, dtype=np.uint8)
    in_ = in_.transpose((2,0,1))
    return in_[np.newaxis, ...]

def load_image( f_path_list):
    im_list=list()
    for path in f_path_list:
        im_list.append(load_image_single(path))
    image_tensor_to_return=np.squeeze(np.array(im_list))
    return image_tensor_to_return
    
def load_mat( f_path_list):
    mat_list=list()
    for path in f_path_list:
        mat_list.append(loadmat(path)['prob_annot'])
    image_tensor_to_return=np.squeeze(np.array(mat_list))
    return image_tensor_to_return
    
def load_label_single(f_path):
    label=Image.open(f_path)
    width_side = label.size[0]
    new_h=width_side/2
    label = label.crop(
        (
            0,
            label.size[1]/2-new_h/2,
            width_side,
            label.size[1]/2+new_h/2
        )
    )
    label = label.resize( (image_size[0],image_size[1]),Image.NEAREST)
    label_return=np.array(label, dtype=np.uint8)
    return label_return[np.newaxis, ...]

def load_label(f_path_list):
    im_list=list()
    for path in f_path_list:
        im_list.append(load_label_single(path))
    label_tensor_to_return=np.squeeze(np.array(im_list))
    if len(label_tensor_to_return.shape) == 2:
        label_tensor_to_return=label_tensor_to_return[ np.newaxis,...]
    return label_tensor_to_return

#load data
# config
from os import listdir,walk
from os.path import isfile, join

class image_label_segment_generator(object):
    def __init__(self, im_list,seg_path):
        self.im_list = im_list
        self.seg_path = seg_path

    def __len__(self):
        return(len(self.im_list))
        
    def __getitem__(self, key):
        im_name_list=[self.im_list[j].split('/')[-1] for j in key]
        seg_list_to_load=[self.seg_path+i for i in im_name_list]

        loaded_label=load_label(seg_list_to_load)
        loaded_im=load_image([self.im_list[j] for j in key])
        return (loaded_im,loaded_label)

synthia_im_path='./data/Image/SYNTHIA/train/'
synthia_im_file_list = [y for x in walk(synthia_im_path) for y in glob(join(x[0], '*.png'))]
val_synthia_im_list=synthia_im_file_list[::30]
train_synthia_im_list=synthia_im_file_list
del train_synthia_im_list[::30]
train_synthia_generator=image_label_segment_generator(train_synthia_im_list,'./data/segmentation_annotation/SYNTHIA/GT/parsed_LABELS/')

cityscape_val_im_path='./data/Image/CityScape/train/'
cityscape_val_im_list = [y for x in walk(cityscape_val_im_path) for y in glob(join(x[0], '*.png'))]
cityscape_val_im_list=cityscape_val_im_list[::5]
val_synthia_generator=image_label_segment_generator(cityscape_val_im_list,'./data/segmentation_annotation/Parsed_CityScape/train/')

cityscape_test_im_path='./data/Image/CityScape/val/'
cityscape_test_im_list=[f for f in listdir(cityscape_test_im_path) if isfile(join(cityscape_test_im_path, f))]
test_cityscape_generator=image_label_segment_generator(cityscape_test_im_list,'./data/segmentation_annotation/Parsed_CityScape/val/')

class image_layout_segment_generator(object):
    def __init__(self, im_list,SP_map_list,SP_annot_list,mat_list):
        self.im_list = im_list
        self.SP_map_list = SP_map_list
        self.SP_annot_list = SP_annot_list
        self.mat_list=mat_list

    def __len__(self):
        return(len(self.im_list))
        
    def __getitem__(self, key):
        seg_name_list=[self.im_list[j] for j in key]
        SP_map_name_list=[self.SP_map_list[j] for j in key]
        SP_annot_name_list=[self.SP_annot_list[j] for j in key]
        mat_name_list=[self.mat_list[j] for j in key]

        loaded_im=load_image(seg_name_list)
        SP_map=load_label(SP_map_name_list)
        SP_annot=load_label(SP_annot_name_list)
        loaded_mat=load_mat(mat_name_list)
        return loaded_im,SP_map,SP_annot,loaded_mat
        
cityscape_im_path='./data/Image/CityScape/'
cityscape_SP_map_path='./data/SP_labels/Parsed_CityScape/'
cityscape_SP_annot_path='./data/SP_landmark/Parsed_CityScape/'
cityscape_mat_path='./data/label_distribution/Parsed_CityScape_inception/'


cityscape_im_list = [y for x in walk(cityscape_im_path) for y in glob(join(x[0], '*.png'))]
aux_data_list=[x.split('/')[-3]+'/'+x.split('/')[-1] for x in cityscape_im_list]
cityscape_SP_map_list = [cityscape_SP_map_path+x[0:-4]+'.png' for x in aux_data_list]
cityscape_SP_annot_list = [cityscape_SP_annot_path+x[0:-4]+'.png' for x in aux_data_list]
cityscape_mat_list = [cityscape_mat_path+x[0:-4]+'.mat' for x in aux_data_list]

cityscape_im_generator=image_layout_segment_generator(cityscape_im_list,cityscape_SP_map_list,cityscape_SP_annot_list,cityscape_mat_list)
