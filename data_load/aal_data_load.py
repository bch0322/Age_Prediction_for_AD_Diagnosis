import torch.utils.data as data
import numpy as np
import torch
import os
import setting as st
import nibabel as nib
import utils
import pickle
from data_load import data_load as DL

# x_range = st.x_range
# y_range = st.y_range
# z_range = st.z_range
# x_size = x_range[1] - x_range[0]
# y_size = y_range[1] - y_range[0]
# z_size = z_range[1] - z_range[0]

def Prepare_data_GM():
    """load the GM data 256 """
    """ GM """
    list_dir_all_modality = []
    GM_sub_list = []
    # included_file_name_GM = []
    included_file_name_GM = ['gm']
    for cnt, dir in enumerate(st.dir_list):
        GM_sub_list.append(utils.search_in_whole_subdir(file_dir=st.orig_data_dir, sub_dir=dir, n_file=included_file_name_GM, n_ext='.img'))
        print(len(GM_sub_list[cnt]))

    """
    AD : 229
    MCI : 403
    AD : 198
    sMCI : 214
    pMCI : 160
    """
    count = 0
    list_MCI = []
    for i in range(len(GM_sub_list[1])):
        cur_count = count
        for j in range(len(GM_sub_list[3])):
            if GM_sub_list[1][i][-25:-14] == GM_sub_list[3][j][-25:-14]:
                count +=1
                list_MCI.append(GM_sub_list[1][i])
        for k in range(len(GM_sub_list[4])):
            if GM_sub_list[1][i][-25:-14] == GM_sub_list[4][k][-25:-14]:
                count +=1
                list_MCI.append(GM_sub_list[1][i])
        if cur_count == count :
            print(GM_sub_list[1][i])

    GM_sub_list.pop(1)
    GM_sub_list.insert(1, list_MCI)

    list_dir_all_modality.append(GM_sub_list)

    """ allocate the memory  """
    list_image_memalloc = []
    list_age_memallow = []
    list_MMSE_memallow = []

    """ the # of the subject depending on the disease label """
    n_NC_subjects = len(list_dir_all_modality[0][0])
    n_MCI_subjects = len(list_dir_all_modality[0][1])
    n_AD_subjects = len(list_dir_all_modality[0][2])
    n_sMCI_subjects = len(list_dir_all_modality[0][3])
    n_pMCI_subjects = len(list_dir_all_modality[0][4])

    list_n_subjects = [n_NC_subjects, n_MCI_subjects, n_AD_subjects, n_sMCI_subjects, n_pMCI_subjects]

    for i in range (len(st.list_class_type)):
        list_image_memalloc.append(np.memmap(filename=st.ADNI_fold_image_path[i], mode="w+", shape=(list_n_subjects[i], st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.uint8))
        list_age_memallow.append(np.memmap(filename=st.ADNI_fold_age_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float32))
        list_MMSE_memallow.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float32))

    """ save the data """
    for i_modality in range (len(list_dir_all_modality)):
        for j_class in range (len(list_dir_all_modality[i_modality])):
            for k in range(len(list_dir_all_modality[i_modality][j_class])):
                tmp_dir_file = list_dir_all_modality[i_modality][j_class][k]
                print(tmp_dir_file[-24:-14])
                print("modality: {}, class: {}, n_sample: {}".format(i_modality, j_class, k))
                tmp_img = np.squeeze(nib.load(tmp_dir_file).get_data())[st.x_range[0] : st.x_range[1], st.y_range[0] : st.y_range[1], st.z_range[0]: st.z_range[1]]
                utils.save_numpy_to_2D_img(tmp_img, save_dir='./plot_img', file_name='/' + tmp_dir_file[-24:-14] + '_sample_class{}_{}'.format(j_class, k))
                list_image_memalloc[j_class][k, i_modality, :, :, :] = tmp_img
