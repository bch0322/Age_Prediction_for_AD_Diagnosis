import pickle
import numpy as np
import setting as st
import pandas as pd
import utils as ut
import os

def Prepare_data_1():
    """ load data jsy processed """
    dat_dir = st.orig_data_dir + '/data.npy'
    cls_dir = st.orig_data_dir + '/label.npy'
    # age_dir = st.orig_data_dir + '/adni_age.npy'
    # id_dir = st.orig_data_dir + '/adni_id.npy'

    adni_dat = np.load(dat_dir, mmap_mode='r')
    adni_cls = np.load(cls_dir, mmap_mode='r')
    # adni_age = np.load(age_dir, mmap_mode='r')
    # adni_id = np.load(id_dir, mmap_mode='r')

    # t_adni_cls = adni_cls

    """ allocation memory """
    list_image_memalloc = []
    list_age_memallow = []
    list_MMSE_memallow = []


    """ the # of the subject depending on the disease label """
    unique, counts = np.unique(adni_cls, return_counts=True)

    n_NC_subjects = counts[0]
    n_MCI_subjects = counts[1]
    n_AD_subjects = counts[2]
    list_n_subjects = [n_NC_subjects, n_MCI_subjects, n_AD_subjects]
    # n_sMCI_subjects = list_final_label.count(1)
    # n_pMCI_subjects = list_final_label.count(2)
    # list_n_subjects = [n_NC_subjects, n_MCI_subjects, n_AD_subjects, n_sMCI_subjects, n_pMCI_subjects]

    for i in range (len(st.list_class_type)):
        list_image_memalloc.append(np.memmap(filename=st.ADNI_fold_image_path[i], mode="w+", shape=(list_n_subjects[i], st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.float32))
        list_age_memallow.append(np.memmap(filename=st.ADNI_fold_age_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float32))
        list_MMSE_memallow.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float32))
    #
    """ save the data """
    count_NC = 0
    count_MCI = 0
    count_AD = 0
    count_total_samples = 0
    for j in range(adni_dat.shape[0]):
        print(f'{j}th subject.')
        count_total_samples +=1
        if adni_cls[j] == 0:
            list_image_memalloc[0][count_NC, 0, :, :, :]= np.squeeze(adni_dat[j])
            # list_age_memallow[0][count_NC] = np.squeeze(adni_age[j])
            count_NC += 1

        elif adni_cls[j] == 1:
            list_image_memalloc[1][count_MCI, 0, :, :, :]= np.squeeze(adni_dat[j])
            # list_age_memallow[1][count_MCI] = np.squeeze(adni_age[j])
            count_MCI += 1

        elif adni_cls[j] == 2:
            list_image_memalloc[2][count_AD, 0, :, :, :]= np.squeeze(adni_dat[j])
            # list_age_memallow[2][count_AD] = np.squeeze(adni_age[j])
            count_AD += 1

    print("count nc : " + str(count_NC)) # 284
    print("count mci : " + str(count_MCI)) # 374
    print("count ad : " + str(count_AD))  # 329