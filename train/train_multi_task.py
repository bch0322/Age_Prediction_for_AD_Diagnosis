import os
import sys
import utils as ut
from torch.autograd import Variable
import torch
import torch.nn as nn
import setting as st
import setting_2 as fst
from data_load import data_load as DL
import numpy as np


def train(config, fold, model, dict_loader, optimizer, scheduler, list_dir_save_model, dir_pyplot, Validation=True, Test_flag = True):

    train_loader = dict_loader['train']
    val_loader = dict_loader['val']
    test_loader = dict_loader['test']

    """ loss """
    # criterion_cls = nn.CrossEntropyLoss()
    # criterion_cls = ut.FocalLoss(gamma=st.focal_gamma, alpha=st.focal_alpha, size_average=True)
    # kdloss = ut.KDLoss(4.0)
    criterion_KL = nn.KLDivLoss(reduction="sum")
    criterion_cls = nn.BCELoss()
    criterion_L1 = nn.L1Loss(reduction='sum').cuda()
    # criterion_L2 = nn.MSELoss(reduction='mean').cuda()
    # criterion_gdl = gdl_loss(pNorm=2).cuda()

    EMS = ut.eval_metric_storage()
    list_selected_EMS = []
    list_ES = []
    for i_tmp in range(len(st.list_standard_eval_dir)):
        list_selected_EMS.append(ut.eval_selected_metirc_storage())
        list_ES.append(ut.EarlyStopping(delta=0, patience=st.early_stopping_patience, verbose=True))

    loss_list_per_batch = [0] * 5
    loss_tmp_total = 0
    print('training')
    optimizer.zero_grad()
    optimizer.step()

    """ epoch """
    num_data = len(train_loader.dataset)
    for epoch in range(1, config.num_epochs+1):
        scheduler.step()
        print(" ")
        print("---------------  epoch {} ----------------".format(epoch))

        """ print learning rate """
        for param_group in optimizer.param_groups:
            print('current LR : {}'.format(param_group['lr']))

        """ batch """
        for i, data_batch in enumerate(train_loader):
            # start = time.time()
            model.train()
            with torch.no_grad():
                """ input"""
                datas = Variable(data_batch['data'].float()).cuda()
                # labels = Variable(data_batch['label'].long()).cuda()
                labels = Variable(data_batch['label'].float()).cuda()
                labels_age = Variable(data_batch['age'].float()).cuda()
                labels_MMSE = Variable(data_batch['MMSE'].float()).cuda()

                """ encode age label """
                # labels_age = ((labels_age - 50) * 10).int()
                # age_encode = torch.zeros(labels_age.size(0), 9).cuda()
                # for tmp_i in range(9):
                #     age_encode[:, tmp_i] = labels_age % 2
                #     labels_age = labels_age // 2
                # age_encode_label = age_encode.clone()

                """ data augmentation """
                ##TODO : flip
                # flip_flag_list = np.random.normal(size=datas.shape[0])>0
                # datas[flip_flag_list] = datas[flip_flag_list].flip(-3)

                ##TODO : translation, cropping
                dict_result = ut.data_augmentation(datas=datas, cur_epoch=epoch)
                datas = dict_result['datas']
                translation_list = dict_result['translation_list']
                # aug_dict_result = ut.data_augmentation(datas=aug_datas, cur_epoch=epoch)
                # aug_datas = aug_dict_result['datas']

                """ minmax norm"""
                if st.list_data_norm_type[st.data_norm_type_num] == 'minmax':
                    tmp_datas = datas.view(datas.size(0), -1)
                    tmp_datas -= tmp_datas.min(1, keepdim=True)[0]
                    tmp_datas /= tmp_datas.max(1, keepdim=True)[0]
                    datas = tmp_datas.view_as(datas)

                """ gaussain noise """
                # Gaussian_dist = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.01]))
                # Gaussian_dist = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), scale=torch.FloatTensor(1).uniform_(0, 0.01))
                # Gaussian_noise = Gaussian_dist.sample(datas.size()).squeeze(-1)
                # datas = datas + Gaussian_noise.cuda()

            """ forward propagation """
            dict_result = model(datas, labels_age.clone())
            output_1 = dict_result['logits']
            output_2 = dict_result['Aux_logits']
            output_3 = dict_result['logitMap']
            output_4 = dict_result['l1_norm']

            #
            loss_list_1 = []
            count_loss = 0
            if fst.flag_loss_1 == True:
                # s_labels = labels
                s_labels = ut.smooth_one_hot(labels, config.num_classes, smoothing=st.smoothing_img)
                loss_2 = criterion_cls(output_1, s_labels) * st.lambda_major[0] / st.iter_to_update
                loss_list_1.append(loss_2)
                loss_list_per_batch[count_loss] += loss_2.data.cpu().numpy()
                if (EMS.total_train_iter + 1) % st.iter_to_update == 0:
                    EMS.train_aux_loss[count_loss].append(loss_list_per_batch[count_loss])
                    loss_list_per_batch[count_loss] = 0
                count_loss += 1

            if fst.flag_loss_2 == True:
                s_labels = labels_age
                loss_list_2 = []
                for i_disease in range(len(st.list_selected_for_train)):
                    if output_2[0][(labels == i_disease)].nelement() != 0:
                        loss_list_2.append(
                            criterion_L1(
                                output_2[0][(labels == i_disease)].squeeze(), s_labels[(labels == i_disease)].squeeze())
                                           * st.list_selected_lambdas_at_loss[i_disease]
                                           / st.iter_to_update
                            )
                loss_2 = sum(loss_list_2) / output_2[0].size(0)
                loss_list_1.append(loss_2)
                loss_list_per_batch[count_loss] += loss_2.data.cpu().numpy()
                if (EMS.total_train_iter + 1) % st.iter_to_update == 0:
                    EMS.train_aux_loss[count_loss].append(loss_list_per_batch[count_loss])
                    loss_list_per_batch[count_loss] = 0
                count_loss += 1

            loss = sum(loss_list_1)
            loss.backward()
            torch.cuda.empty_cache()
            loss_tmp_total += loss.data.cpu().numpy()

            #TODO :  optimize the model param
            if (EMS.total_train_iter + 1) % st.iter_to_update == 0:
                optimizer.step()
                optimizer.zero_grad()

                """ pyplot """
                EMS.total_train_step += 1
                EMS.train_step.append(EMS.total_train_step)
                EMS.train_loss.append(loss_tmp_total)

                """ print the train loss and tensorboard"""
                if (EMS.total_train_step) % 10 == 0 :
                    # print('time : ', time.time() - start)
                    print('Epoch [%d/%d], Step [%d/%d],  Loss: %.4f' %(epoch, config.num_epochs, (i + 1), (num_data // (config.batch_size)), loss_tmp_total))
                loss_tmp_total = 0

            EMS.total_train_iter += 1
            # scheduler.step(epoch + i / len(train_loader))

        """ val """
        if Validation == True:
            print("------------------  val  --------------------------")
            dict_result = ut.eval_multi_task_model(config, fold, val_loader, model, criterion_cls, criterion_L1)
            val_loss = dict_result['Loss']
            acc = dict_result['Acc']
            auc = dict_result['AUC']
            loss_age = dict_result['loss_age']

            print('Fold : %d, Epoch [%d/%d] val Loss = %f val Acc = %f val loss age = %f' % (fold, epoch, config.num_epochs, val_loss, acc, loss_age))

            """ save the metric """
            EMS.dict_val_metric['val_loss'].append(val_loss)
            EMS.dict_val_metric['val_acc'].append(acc)
            EMS.dict_val_metric['val_auc'].append(auc)
            EMS.dict_val_metric['val_loss_age'].append(loss_age)
            EMS.val_step.append(EMS.total_train_step)

            n_stacking_loss_for_selection = 5
            if len(EMS.dict_val_metric['val_loss_queue']) > n_stacking_loss_for_selection:
                EMS.dict_val_metric['val_loss_queue'].popleft()
            EMS.dict_val_metric['val_loss_queue'].append(val_loss)
            EMS.dict_val_metric['val_mean_loss'].append(np.mean(EMS.dict_val_metric['val_loss_queue']))

            """ save model """
            for i_tmp in range(len(list_selected_EMS)):
                save_flag = ut.model_save_through_validation(fold, epoch, EMS=EMS,
                                                             selected_EMS=list_selected_EMS[i_tmp],
                                                             ES=list_ES[i_tmp],
                                                             model=model,
                                                             dir_save_model=list_dir_save_model[i_tmp],
                                                             metric_1=st.list_standard_eval[i_tmp], metric_2='',
                                                             save_flag=False)


        if Test_flag== True:
            print("------------------  test _ test dataset  --------------------------")
            """ load data """
            dict_result = ut.eval_multi_task_model(config, fold, test_loader, model, criterion_cls, criterion_L1)
            acc = dict_result['Acc']
            test_loss = dict_result['Loss']
            loss_age = dict_result['loss_age']

            """ pyplot """
            EMS.test_acc.append(acc)
            EMS.test_loss.append(test_loss)
            EMS.test_loss_age.append(loss_age)
            EMS.test_step.append(EMS.total_train_step)

            print('number of test samples : {}'.format(len(test_loader.dataset)))
            print('Fold : %d, Epoch [%d/%d] test Loss = %f test Acc = %f test loss age = %f' % (fold, epoch, config.num_epochs, test_loss, acc, loss_age))


        """ learning rate decay"""
        EMS.LR.append(optimizer.param_groups[0]['lr'])
        # scheduler.step()
        # scheduler.step(val_loss)

        """ plot the chat """
        if epoch % 1 == 0:
            ut.plot_training_info_1(fold, dir_pyplot, EMS,  flag='percentile', flag_match=False)

        ##TODO : early stop only if all of metric has been stopped
        tmp_count = 0
        for i in range(len(list_ES)):
            if list_ES[i].early_stop == True:
                tmp_count += 1
        if tmp_count == len(list_ES):
            break

    """ release the model """
    del model, EMS
    torch.cuda.empty_cache()
