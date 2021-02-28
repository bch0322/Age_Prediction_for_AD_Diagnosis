import numpy as np
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import setting as st
import setting_2 as fst
from data_load import data_load as DL
import torch
import torch.nn as nn
import utils as ut

def get_heatmap_2class(config, fold, model, dir_to_load, dir_heatmap):
    """ free all GPU memory """
    torch.cuda.empty_cache()

    """ loss """
    criterion = nn.L1Loss()

    """ load the fold list for test """
    list_test_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_test,
                                                 flag_tr_val_te='test')
    test_loader = DL.convert_Dloader_3(config.v_batch_size, list_test_data[0], list_test_data[1], list_test_data[2],
                                       list_test_data[3], is_training=False, num_workers=0, shuffle=False)


    """ load the model """
    model_dir = ut.model_dir_to_load(fold, dir_to_load)
    if model_dir !=None:
        model.load_state_dict(torch.load(model_dir))
    model.eval()

    """ param for accuracy """
    test_batchnum = 0

    """ eval """
    patch_size = st.patch_size
    stride_between_patches = st.patch_stride
    count = 0



    with torch.no_grad():
        for datas, labels, alabel, mlabel in test_loader:
            datas_backup=datas
            count += 1

            """ prediction """
            datas_pred = Variable(datas).cuda()
            dict_result = model(datas_pred)
            output_logit = dict_result['logits']
            prob = nn.Softmax(dim=1)(output_logit)
            pred = prob.argmax(dim=1, keepdim=True)  # batch, 1


            """ padding """
            logit_map = np.zeros_like(datas)
            attn_1_map = np.zeros_like(datas)
            attn_2_map = np.zeros_like(datas)
            final_evidence_map = np.zeros_like(datas)

            """ padding """
            m = nn.ConstantPad3d(patch_size//2, 0)
            datas = m(datas)

            """ loop as much as the size of strides """
            for i in range(stride_between_patches):
                print("i : {0}".format(i))
                for j in range(stride_between_patches):
                    # print("j : {0}".format(j))
                    for k in range(stride_between_patches):

                        """ input"""
                        data = Variable(datas[:, :, i:, j:, k:]).cuda()
                        labels = Variable(labels.long()).cuda()

                        """ run classification model """
                        dict_result = model(data)
                        logitMap = dict_result['logitMap']
                        attn_1 = dict_result['attn_1']
                        attn_2 = dict_result['attn_2']
                        final_evidence = dict_result['final_evidence']


                        shape = logitMap.shape
                        for a in range(shape[-3]):
                            for b in range(shape[-2]):
                                for c in range(shape[-1]):
                                    if logitMap is not None:
                                        logit_map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            logitMap[:, pred[0].data.cpu().numpy()[0], a, b, c].data.cpu().numpy()

                                    if attn_1 is not None:
                                        attn_1_map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            attn_1[:, 0, a, b, c].data.cpu().numpy()
                                    if attn_2 is not None:
                                        attn_2_map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            attn_2[:, 0, a, b, c].data.cpu().numpy()
                                    if final_evidence is not None:
                                        final_evidence_map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            final_evidence[:, pred[0].data.cpu().numpy()[0], a, b, c].data.cpu().numpy()
                            # logit_list.append(test_output_1.cpu().numpy())

            torch.cuda.empty_cache()
            print("finished a sample!")
            for sample in range(logit_map.shape[0]):
                tmp_save_dir = dir_heatmap + '/fold_{0}'.format(fold,

                                                                                )
                ut.make_dir(dir=tmp_save_dir, flag_rm=False)
                ut.save_featureMap_tensor(datas_backup[sample][0], tmp_save_dir, 'input_{0}_gt_{1}_pred_{2}'.format(
                count,
                st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                ))

                if logitMap is not None:
                    ut.save_featureMap_numpy(logit_map[sample][0], tmp_save_dir, 'logit_map_{0}_gt_{1}_pred_{2}'.format(
                count,
                st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = logit_map[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/1_logit_map_{}'.format(count), fig_title='Original Logit Map', thresh=0.2, percentile=1)

                if attn_1 is not None:
                    ut.save_featureMap_numpy(attn_1_map[sample][0], tmp_save_dir, 'attn_1_map_{0}_gt_{1}_pred_{2}'.format(
                count,
                st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = attn_1_map[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/2_attn_map_1_{}'.format(count), fig_title='Attention Map 1', thresh=0.2, percentile=1)

                if attn_2 is not None:
                    ut.save_featureMap_numpy(attn_2_map[sample][0], tmp_save_dir, 'attn_2_map_{0}_gt_{1}_pred_{2}'.format(
                count,
                st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = attn_2_map[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/2_attn_map_2_{}'.format(count), fig_title='Attention Map 2', thresh=0.2, percentile=1)

                if final_evidence is not None:
                    ut.save_featureMap_numpy(final_evidence_map[sample][0], tmp_save_dir, 'final_evidence_map_{0}_gt_{1}_pred_{2}'.format(
                count,
                st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = final_evidence_map[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/3_final_evidence_{}'.format(count), fig_title='Final Evidence', thresh=0.2, percentile=1)

def get_multi_heatmap_2class(config, fold, model, dir_to_load, dir_heatmap):
    """ free all GPU memory """
    torch.cuda.empty_cache()

    """ loss """
    criterion = nn.L1Loss()

    """ load the fold list for test """
    list_test_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_test,
                                                 flag_tr_val_te='test')
    test_loader = DL.convert_Dloader_3(config.v_batch_size, list_test_data[0], list_test_data[1], list_test_data[2],
                                       list_test_data[3], is_training=False, num_workers=0, shuffle=False)


    """ load the model """
    model_dir = ut.model_dir_to_load(fold, dir_to_load)
    if model_dir !=None:
        model.load_state_dict(torch.load(model_dir))
    model.eval()

    """ param for accuracy """
    test_batchnum = 0

    """ eval """
    patch_size_1 = 9
    patch_size_2 = 17
    patch_size_3 = 33
    stride_between_patches = st.patch_stride
    count = 0

    with torch.no_grad():
        for datas, labels, alabel, mlabel in test_loader:
            datas_backup=datas
            count += 1
            datas_pred = Variable(datas).cuda()
            dict_result = model(datas_pred)
            output_logit = dict_result['logits']
            prob = nn.Softmax(dim=1)(output_logit)
            pred = prob.argmax(dim=1, keepdim=True) # batch, 1

            """ padding """
            logit_map = np.zeros_like(datas)
            attn_1_map = np.zeros_like(datas)
            attn_2_map = np.zeros_like(datas)
            attn_3_map = np.zeros_like(datas)
            final_evidence_map_1 = np.zeros_like(datas)
            final_evidence_map_2 = np.zeros_like(datas)
            final_evidence_map_3 = np.zeros_like(datas)

            """ padding """
            m_1 = nn.ConstantPad3d(patch_size_1 // 2, 0)
            datas_1 = m_1(datas)
            m_2 = nn.ConstantPad3d(patch_size_2 // 2, 0)
            datas_2 = m_2(datas)
            m_3 = nn.ConstantPad3d(patch_size_3 // 2, 0)
            datas_3 = m_3(datas)

            """ loop as much as the size of strides """
            for i in range(stride_between_patches):
                print("i : {0}".format(i))
                for j in range(stride_between_patches):
                    # print("j : {0}".format(j))
                    for k in range(stride_between_patches):
                        test_batchnum += 1

                        """ input"""
                        data_1 = Variable(datas_1[:, :, i:, j:, k:]).cuda()
                        dict_result = model(data_1)
                        attn_1 = dict_result['attn_1']  # 1, 1, 25, 30, 24
                        final_evidence_a = dict_result['final_evidence_a']

                        data_2 = Variable(datas_2[:, :, i:, j:, k:]).cuda()
                        dict_result = model(data_2)
                        attn_2 = dict_result['attn_2']  # 1, 1, 24, 29, 23
                        final_evidence_b = dict_result['final_evidence_b']

                        data_3 = Variable(datas_3[:, :, i:, j:, k:]).cuda()
                        dict_result = model(data_3)
                        attn_3 = dict_result['attn_3'] # 1, 1, 22, 27, 21
                        final_evidence_c = dict_result['final_evidence_c']

                        shape = final_evidence_a.shape
                        for a in range(shape[-3]):
                            for b in range(shape[-2]):
                                for c in range(shape[-1]):
                                    if attn_1 is not None:
                                        tmp_index_2 = [a, b ,c]
                                        attn_1_map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            attn_1[:, 0, tmp_index_2[0], tmp_index_2[1], tmp_index_2[2]].data.cpu().numpy()

                                    if final_evidence_a is not None:
                                        tmp_index_2 = [a, b ,c]
                                        final_evidence_map_1[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            final_evidence_a[:, 0, tmp_index_2[0], tmp_index_2[1], tmp_index_2[2]].data.cpu().numpy()

                        shape = final_evidence_b.shape
                        for a in range(shape[-3]):
                            for b in range(shape[-2]):
                                for c in range(shape[-1]):
                                    if attn_2 is not None:
                                        tmp_index_2 = [a , b, c ]
                                        attn_2_map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            attn_2[:, 0, tmp_index_2[0], tmp_index_2[1], tmp_index_2[2]].data.cpu().numpy()

                                    if final_evidence_b is not None:
                                        tmp_index_2 = [a, b, c]
                                        final_evidence_map_2[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            final_evidence_b[:, 0, tmp_index_2[0], tmp_index_2[1], tmp_index_2[2]].data.cpu().numpy()

                        shape = final_evidence_c.shape
                        for a in range(shape[-3]):
                            for b in range(shape[-2]):
                                for c in range(shape[-1]):
                                    if attn_3 is not None:
                                        tmp_index_2 = [a, b, c]
                                        attn_3_map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            attn_3[:, 0, tmp_index_2[0], tmp_index_2[1], tmp_index_2[2]].data.cpu().numpy()
                                    if final_evidence_c is not None:
                                        tmp_index_2 = [a, b, c]
                                        final_evidence_map_3[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            final_evidence_c[:, 0, tmp_index_2[0], tmp_index_2[1], tmp_index_2[2]].data.cpu().numpy()

            torch.cuda.empty_cache()
            print("finished a sample!")
            for sample in range(logit_map.shape[0]):
                tmp_save_dir = dir_heatmap + '/fold_{0}'.format(fold)
                ut.make_dir(dir=tmp_save_dir, flag_rm=False)
                ut.save_featureMap_tensor(datas_backup[sample][0], tmp_save_dir, 'input_{0}_gt_{1}_pred_{2}'.format(
                    count,
                    st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                    st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                    ))

                if attn_1 is not None:
                    ut.save_featureMap_numpy(attn_1_map[sample][0], tmp_save_dir, 'attn_1_map_{0}_gt_{1}_pred_{2}'.format(
                    count,
                    st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                    st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                    ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = attn_1_map[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/2_attn_map_1_{}'.format(count), fig_title='Attention Map 1', thresh=0.2, percentile=1)

                if attn_2 is not None:
                    ut.save_featureMap_numpy(attn_2_map[sample][0], tmp_save_dir, 'attn_2_map_{0}_gt_{1}_pred_{2}'.format(
                    count,
                    st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                    st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                    ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = attn_2_map[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/2_attn_map_2_{}'.format(count), fig_title='Attention Map 2', thresh=0.2, percentile=1)


                if attn_3 is not None:
                    ut.save_featureMap_numpy(attn_3_map[sample][0], tmp_save_dir, 'attn_3_map_{0}_gt_{1}_pred_{2}'.format(
                    count,
                    st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                    st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                    ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = attn_3_map[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/2_attn_map_3_{}'.format(count), fig_title='Attention Map 3', thresh=0.2, percentile=1)


                if final_evidence_a is not None:
                    ut.save_featureMap_numpy(final_evidence_map_1[sample][0], tmp_save_dir, 'final_evidence_map_1_{0}_gt_{1}_pred_{2}'.format(
                    count,
                    st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                    st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                    ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = final_evidence_map_1[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/3_final_evidence_1_{}'.format(count), fig_title='Final Evidence 1', thresh=0.2, percentile=1)


                if final_evidence_b is not None:
                    ut.save_featureMap_numpy(final_evidence_map_2[sample][0], tmp_save_dir, 'final_evidence_map_2_{0}_gt_{1}_pred_{2}'.format(
                    count,
                    st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                    st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                    ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = final_evidence_map_2[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/3_final_evidence_2_{}'.format(count), fig_title='Final Evidence 2', thresh=0.2, percentile=1)

                if final_evidence_c is not None:
                    ut.save_featureMap_numpy(final_evidence_map_3[sample][0], tmp_save_dir, 'final_evidence_map_3_{0}_gt_{1}_pred_{2}'.format(
                    count,
                    st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                    st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                    ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = final_evidence_map_3[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/3_final_evidence_3_{}'.format(count), fig_title='Final Evidence 3', thresh=0.2, percentile=1)

def get_heatmap_1class(config, fold, model, dir_to_load, dir_heatmap):
    """ free all GPU memory """
    torch.cuda.empty_cache()

    """ loss """
    criterion = nn.L1Loss()

    """ load the fold list for test """
    list_test_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_test,
                                                 flag_tr_val_te='test')
    test_loader = DL.convert_Dloader_3(config.v_batch_size, list_test_data[0], list_test_data[1], list_test_data[2],
                                       list_test_data[3], is_training=False, num_workers=0, shuffle=False)


    """ load the model """
    model_dir = ut.model_dir_to_load(fold, dir_to_load)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    """ param for accuracy """
    test_batchnum = 0

    """ eval """
    patch_size = st.patch_size
    stride_between_patches = st.patch_stride
    count = 0
    with torch.no_grad():
        for datas, labels, alabels, mlabel in test_loader:
            count += 1
            if count < 3 :
                datas_backup=datas

                """ padding """
                pred_Map = np.zeros_like(datas)

                """ padding """
                m = nn.ConstantPad3d(patch_size//2, 0)
                datas = m(datas)

                """ loop as much as the size of strides """
                for i in range(stride_between_patches):
                    print("i : {0}".format(i))
                    for j in range(stride_between_patches):
                        # print("j : {0}".format(j))
                        for k in range(stride_between_patches):
                            test_batchnum += 1

                            """ input"""
                            data = Variable(datas[:, :, i:, j:, k:]).cuda()
                            labels = Variable(labels.long()).cuda()
                            alabels = Variable(alabels.float()).cuda()

                            """ run classification model """
                            dict_result = model(data)
                            predMap = dict_result['predMap']
                            shape = predMap.shape
                            for a in range(shape[-3]):
                                for b in range(shape[-2]):
                                    for c in range(shape[-1]):
                                        if predMap is not None:
                                            pred_Map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                                predMap[:, 0, a, b, c].data.cpu().numpy()
                            # logit_list.append(test_output_1.cpu().numpy())

                torch.cuda.empty_cache()
                print("finished a sample!")
                for sample in range(pred_Map.shape[0]):
                    tmp_save_dir = dir_heatmap + '/fold_{0}'.format(fold)
                    ut.make_dir(dir=tmp_save_dir, flag_rm=False)
                    ut.save_featureMap_tensor(datas_backup[sample][0], tmp_save_dir, 'input_{}'.format(count))

                    if predMap is not None:
                        ut.save_featureMap_numpy(pred_Map[sample][0], tmp_save_dir, 'pred_map_{}'.format(count))
                        orig_img = datas_backup[sample][0]
                        heatmap_img = pred_Map[sample][0]
                        ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/1_logit_map', fig_title='Original Logit Map', thresh=0.2, percentile=1)
