import numpy as np
import torch
import setting as st
import nibabel as nib
import utils
import pandas as pd
import os
import pickle

def Prepare_data_GM_AGE_MMSE():
    """ tadpole dataset """
    tadpole_dir = st.tadpole_dir
    data = pd.read_csv(tadpole_dir)
    data_bl = data[data.VISCODE == 'bl']
    PTID_uniq_tp = np.unique(data.PTID.values)

    # sorting the whole value depending on the PTID and VISCODE
    sorted_data = data.sort_values(by=['PTID', 'VISCODE'])
    np_sorted_data = sorted_data.to_numpy()


    """ label information extraction """
    # label_dir ="/Data/cwkim/New_Preprocessing/label/label.txt"
    label_dir = st.orig_data_dir + "/label/label.txt"
    f = open(label_dir, 'r')
    list_fileName_txt = []
    list_age_txt = []
    list_label_txt = []
    list_PTID_txt = []
    while True:
        line = f.readline()
        if not line:
            break
        # Remove the leading spaces and newline character
        line = line.strip()

        # Split the line into words
        words = line.split(" ")
        assert len(words) == 4

        # print(line)
        if 'ADNI' in words[0]:
            list_fileName_txt.append(words[0])
            list_age_txt.append(float(words[1]))
            list_label_txt.append(int(words[2]))
            list_PTID_txt.append(words[0][10:20])
    f.close()

    """ check whether file is loaded in order"""
    included_file_name_GM = ['ADNI']
    densityMap_dir_list = utils.search_in_whole_subdir(st.orig_data_dir, "", included_file_name_GM, '.nii')
    count_tmp = 0
    list_ordered_label = []
    list_ordered_age = []
    list_ordered_PTID = []
    list_ordered_fileName = []
    for i in range(len(densityMap_dir_list)):
        for j in range(len(list_fileName_txt)):
            if densityMap_dir_list[i] == st.orig_data_dir + '/GM' + '/' + list_fileName_txt[j]:
                list_ordered_age.append(list_age_txt[j])
                list_ordered_label.append(list_label_txt[j])
                list_ordered_PTID.append(list_PTID_txt[j])
                list_ordered_fileName.append(list_fileName_txt[j])
                count_tmp += 1
    print("Out of {}, the matching number with txt is {}.".format(len(densityMap_dir_list), count_tmp))

    """ check the number of the labels each """
    count_NC = list_ordered_label.count(0)
    count_MCI = list_ordered_label.count(1)
    count_sMCI = list_ordered_label.count(2)
    print("count 0 : " + str(count_NC)) # 284
    print("count 1 : " + str(count_MCI)) # 374
    print("count 2 : " + str(count_sMCI)) # 329

    """ 
    matching tp and txt 
    """
    count = 0
    count_SMC = 0
    list_final_PTID = []
    list_final_label = []
    list_final_age = []
    list_final_MMSE = []
    list_final_fileName = []
    for i in range(len(list_ordered_PTID)):
        for j in range(PTID_uniq_tp.shape[0]):
            if list_ordered_PTID[i] in PTID_uniq_tp[j]:
                if np.isnan(data_bl[data.PTID == PTID_uniq_tp[j]].MMSE.values) == False:
                    count += 1
                    list_final_PTID.append(list_ordered_PTID[i])
                    list_final_label.append(list_ordered_label[i])
                    list_final_age.append(list_ordered_age[i])
                    list_final_fileName.append(list_ordered_fileName[i])
                    list_final_MMSE.append(data_bl[data.PTID == PTID_uniq_tp[j]].MMSE.values)
                    if (data_bl[data.PTID == PTID_uniq_tp[j]].DX_bl.values[0]) == 'SMC':
                        count_SMC += 1
    print(len(list_ordered_PTID))
    print(count)
    print("the # of SMC :  {}".format(count_SMC))
    print("PTID : " + str(len(list_final_PTID)))

    np_label = np.array(list_final_label)
    np_label[np_label == 2] = 3
    list_final_label = np_label.tolist()
    print("label : " + str(len(list_final_label)))
    print("label_0 : " + str((list_final_label.count(0))))
    print("label_1 : " + str((list_final_label.count(1))))
    print("label_2 : " + str((list_final_label.count(2))))
    print("label_3 : " + str((list_final_label.count(3))))

    print("age : " + str(len(list_final_age)))
    print("MMSE : " + str(len(list_final_MMSE)))

    """ labeling the every month and check whether they are sMCI or pMCI"""
    # 1 : PTID / 2: month / 9: disease label / 10 : dxchang / 11: age / 24: MMSE / sorted_data.columns[24]
    list_Rev_PTID = []
    list_nan_PTID = []
    i_row = 0
    count_MCI_without_m36 = []
    list_flag_MCI_without_m36 = []
    count_bl_nan_dxchange = 0
    list_true_reverse = []
    while True:
        # i_row += 1

        # index check
        if i_row >= np_sorted_data.shape[0]:
            break

        flag_start = True
        bl_index = 0
        if np_sorted_data[i_row, 2] == 'bl':
            if flag_start == True:
                bl_index = i_row
                flag_start = False

            flag_pMCI_search = False
            pMCI_labeling = False

            if np.isnan(np_sorted_data[i_row, 10]):
                count_bl_nan_dxchange += 1

            if np_sorted_data[i_row, 9] == 'CN':
                flag_MCI_without_m36 = 0

            elif np_sorted_data[i_row, 9] == 'AD':
                flag_MCI_without_m36 = 3
            elif np_sorted_data[i_row, 9] == 'EMCI':
                flag_MCI_without_m36 = 1
                flag_pMCI_search = True

            elif np_sorted_data[i_row, 9] == 'LMCI':
                flag_MCI_without_m36 = 1
                flag_pMCI_search = True
            elif np_sorted_data[i_row, 9] == 'SMC':
                flag_MCI_without_m36 = 0

            while True:
                i_row += 1
                # index check
                if i_row >= np_sorted_data.shape[0]:
                    break

                # if not the current is next PTID
                if np_sorted_data[i_row, 2] == 'bl':
                    break

                # if current bl is MCI and m36 exist in the loop
                if flag_MCI_without_m36 == 1 and np_sorted_data[i_row, 2] == 'm36':
                    flag_MCI_without_m36 = 2

                # check whether the VISCODE is over than m36
                if flag_pMCI_search == True and int(np_sorted_data[i_row, 2][1:]) > 36:
                    flag_pMCI_search = False

                # check whether the VIS
                if flag_pMCI_search == True:
                    if np.isnan(np_sorted_data[i_row, 10]):
                        pass
                    elif np_sorted_data[i_row, 10] == 2:
                        np_sorted_data[i_row, 9] = 'MCI'
                    elif np_sorted_data[i_row, 10] == 3:
                        np_sorted_data[i_row, 9] = 'AD'
                    elif np_sorted_data[i_row, 10] == 5:  # MCI to AD
                        np_sorted_data[i_row, 9] = 'AD'
                        pMCI_labeling = True
                    else: # reversion
                        # for i in range(len(list_final_PTID)):
                        #     if list_final_PTID[i] == np_sorted_data[i_row, 1]:
                        #         list_final_PTID.pop(i)
                        #         list_final_label.pop(i)
                        #         list_final_age.pop(i)
                        #         list_final_MMSE.pop(i)
                        #         list_final_fileName.pop(i)
                        #         list_true_reverse.append(np_sorted_data[i_row, 1])
                        #         break
                        flag_pMCI_search = False

                # converting check
                if np.isnan(np_sorted_data[i_row, 10]):
                    np_sorted_data[i_row, 9] = 'nan'
                    list_nan_PTID.append(np_sorted_data[i_row, 1])
                elif np_sorted_data[i_row, 10] == 1:
                    np_sorted_data[i_row, 9] = 'CN'
                elif np_sorted_data[i_row, 10] == 2:
                    np_sorted_data[i_row, 9] = 'MCI'
                elif np_sorted_data[i_row, 10] == 3:
                    np_sorted_data[i_row, 9] = 'AD'
                elif np_sorted_data[i_row, 10] == 4:  # NL to MCI
                    np_sorted_data[i_row, 9] = 'MCI'
                elif np_sorted_data[i_row, 10] == 5:  # MCI to AD
                    np_sorted_data[i_row, 9] = 'AD'
                elif np_sorted_data[i_row, 10] == 6:  # NL to AD
                    np_sorted_data[i_row, 9] = 'AD'
                else:
                    np_sorted_data[i_row, 9] = 'Rev'
                    list_Rev_PTID.append(np_sorted_data[i_row, 1])
                    for i in range(len(list_final_PTID)):
                        if list_final_PTID[i] == np_sorted_data[i_row, 1]:
                            list_final_PTID.pop(i)
                            list_final_label.pop(i)
                            list_final_age.pop(i)
                            list_final_MMSE.pop(i)
                            list_final_fileName.pop(i)
                            list_true_reverse.append(np_sorted_data[i_row, 1])
                            break
            list_flag_MCI_without_m36.append(flag_MCI_without_m36)
            if pMCI_labeling == True:
                for i in range(len(list_final_label)):
                    if np_sorted_data[bl_index, 1] == list_final_PTID[i]:
                        assert list_final_label[i] == 1
                        list_final_label[i] = 2

    print("unique PTID who have got rev for dxchange : {} / {}".format(np.unique(np.array(list_Rev_PTID)).shape,
                                                                       np.unique((np_sorted_data[:, 1])).shape))
    print("unique PTID who have got nan for dxchange : {} / {}".format(np.unique(np.array(list_nan_PTID)).shape,
                                                                       np.unique((np_sorted_data[:, 1])).shape))
    print("pt who are bl and nan for dxchange simultaneously : {}.".format(count_bl_nan_dxchange))
    print("total : {}".format(len(list_flag_MCI_without_m36)))
    print("MCI_m36 : {0}, {1}, {2}, {3}".format(list_flag_MCI_without_m36.count(0), list_flag_MCI_without_m36.count(1),
                                                list_flag_MCI_without_m36.count(2), list_flag_MCI_without_m36.count(3)))

    print("label : " + str(len(list_final_label)))
    print("label_0 : " + str((list_final_label.count(0))))
    print("label_1 : " + str((list_final_label.count(1))))
    print("label_2 : " + str((list_final_label.count(2))))
    print("label_3 : " + str((list_final_label.count(3))))
    print("true reverse {}".format(np.unique(np.array(list_true_reverse)).shape))
    print('finished')

    """ allocation memory """
    list_image_memalloc = []
    list_age_memallow = []
    list_MMSE_memallow = []

    """ the # of the subject depending on the disease label """

    n_NC_subjects = list_final_label.count(0)
    n_MCI_subjects = list_final_label.count(1) + list_final_label.count(2)
    n_AD_subjects = list_final_label.count(3)
    n_sMCI_subjects = list_final_label.count(1)
    n_pMCI_subjects = list_final_label.count(2)
    list_n_subjects = [n_NC_subjects, n_MCI_subjects, n_AD_subjects, n_sMCI_subjects, n_pMCI_subjects]

    """ NC """
    for i in range (len(st.list_class_type)):
        list_image_memalloc.append(np.memmap(filename=st.ADNI_fold_image_path[i], mode="w+", shape=(list_n_subjects[i], st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.float64))
        list_age_memallow.append(np.memmap(filename=st.ADNI_fold_age_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float64))
        list_MMSE_memallow.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float64))

    """ save the data """
    count_NC = 0
    count_MCI = 0
    count_sMCI = 0
    count_pMCI = 0
    count_AD = 0
    count_total_samples = 0
    for cnt, p in enumerate(densityMap_dir_list):
        print(cnt)
        for j in range(len(list_final_fileName)):
            if densityMap_dir_list[cnt] == st.orig_data_dir + '/GM' + '/' + list_final_fileName[j]:
                count_total_samples +=1
                if list_final_label[j] == 0:
                    list_image_memalloc[0][count_NC, 0, :, :, :]= np.squeeze(nib.load(p).get_data())
                    list_age_memallow[0][count_NC] = np.squeeze(list_final_age[j])
                    list_MMSE_memallow[0][count_NC] = np.squeeze(list_final_MMSE[j])
                    count_NC += 1

                elif list_final_label[j] == 1:
                    list_image_memalloc[3][count_sMCI, 0, :, :, :]= np.squeeze(nib.load(p).get_data())
                    list_age_memallow[3][count_sMCI] = np.squeeze(list_final_age[j])
                    list_MMSE_memallow[3][count_sMCI] = np.squeeze(list_final_MMSE[j])
                    count_sMCI += 1

                elif list_final_label[j] == 2:
                    list_image_memalloc[4][count_pMCI, 0, :, :, :] = np.squeeze(nib.load(p).get_data())
                    list_age_memallow[4][count_pMCI] = np.squeeze(list_final_age[j])
                    list_MMSE_memallow[4][count_pMCI] = np.squeeze(list_final_MMSE[j])
                    count_pMCI += 1

                elif list_final_label[j] == 3:
                    list_image_memalloc[2][count_AD, 0, :, :, :]= np.squeeze(nib.load(p).get_data())
                    list_age_memallow[2][count_AD] = np.squeeze(list_final_age[j])
                    list_MMSE_memallow[2][count_AD] = np.squeeze(list_final_MMSE[j])
                    count_AD += 1

                """ MCI which should be gathered with sMCI and pMCI """
                if list_final_label[j] == 1 or list_final_label[j] == 2:
                    list_image_memalloc[1][count_MCI, 0, :, :, :]= np.squeeze(nib.load(p).get_data())
                    list_age_memallow[1][count_MCI] = np.squeeze(list_final_age[j])
                    list_MMSE_memallow[1][count_MCI] = np.squeeze(list_final_MMSE[j])
                    count_MCI += 1


    print("count nc : " + str(count_NC)) # 284
    print("count mci : " + str(count_MCI)) # 374
    print("count smci : " + str(count_sMCI)) # 329
    print("count pmci : " + str(count_pMCI))  # 329
    assert count_MCI == count_sMCI + count_pMCI
    print("count ad : " + str(count_AD))  # 329


def Prepare_data_GM_age_others(dataset = 'ABIDE'):

    """ data dir check """
    included_file_name_GM = [dataset]
    GM_sub_list = utils.search_in_whole_subdir(st.orig_data_dir, "", included_file_name_GM, '.nii')

    """ label information extraction """
    # label_dir ="/Data/cwkim/New_Preprocessing/label/label.txt"
    label_dir = st.orig_data_dir + "/label/label.txt"
    f = open(label_dir, 'r')
    list_dir = []
    list_age = []
    list_label = []

    while True:
        line = f.readline()
        if not line:
            break
        # Remove the leading spaces and newline character
        line = line.strip()

        # Convert the characters in line to
        # lowercase to avoid case mismatch
        # line = line.lower()

        # Split the line into words
        words = line.split(" ")
        assert len(words) == 4

        # print(line)
        list_dir.append(words[0])
        list_age.append(float(words[1]))
        list_label.append(int(words[2]))
    f.close()

    """ check whether file is loaded in order"""
    count_tmp = 0
    selected_label = []
    selected_age = []
    selected_dir = []
    for i in range(len(GM_sub_list)):
        for j in range (len(list_dir)):
            if GM_sub_list[i] == st.orig_data_dir + '/GM' + '/' + list_dir[j]:
                selected_age.append(list_age[j])
                selected_label.append(list_label[j])
                selected_dir.append(GM_sub_list[i])
                count_tmp += 1
    print(count_tmp)

    count_0 = 0
    count_1 = 0
    count_2 = 0
    for i in range(len(selected_dir)):
        if selected_label[i] == 0:
            count_0 += 1
        elif selected_label[i] == 1:
            count_1 += 1
        elif selected_label[i] == 2:
            count_2 += 1
    print("count 0 : " + str(count_0)) # 284
    print("count 1 : " + str(count_1)) # 374
    print("count 2 : " + str(count_2)) # 329


    """ allocation memory """
    NC_img_dat = np.memmap(filename=st.orig_npy_dir + "/"+dataset+'_NC_raw.npy', mode="w+", shape=(count_0, st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.float64)
    NC_age_dat = np.memmap(filename=st.orig_npy_dir + "/"+dataset+'_NC_age.npy', mode="w+", shape=(count_0), dtype=np.float64)

    """ save the data """
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for cnt, p in enumerate(selected_dir):
        print(cnt)
        if selected_label[cnt] == 0:
            NC_img_dat[count_0, 0, :, :, :]= np.squeeze(nib.load(p).get_data())
            NC_age_dat[count_0] = np.squeeze(selected_age[cnt])
            count_0 += 1

    print("count 0 : " + str(count_0)) # 284
    print("count 1 : " + str(count_1)) # 374
    print("count 2 : " + str(count_2)) # 329