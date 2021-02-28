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


def train(config, fold, model_1, model_2, dict_loader, optimizer, scheduler, list_dir_save_model, dir_pyplot, Validation=True, Test_flag = True):

    train_loader = dict_loader['train']
    val_loader = dict_loader['val']
    test_loader = dict_loader['test']

    """ loss """
    # criterion_cls = nn.CrossEntropyLoss()
    # criterion_cls = ut.FocalLoss(gamma=st.focal_gamma, alpha=st.focal_alpha, size_average=True)
    criterion_cls = nn.BCELoss()
    # criterion = nn.L1Loss(reduction='mean').cuda()
    criterion = nn.MSELoss(reduction='mean').cuda()
    # criterion_gdl = gdl_loss(pNorm=2).cuda()

    EMS = ut.eval_metric_storage()
    list_selected_EMS = []
    list_ES = []
    for i_tmp in range(len(st.list_standard_eval_dir)):
        list_selected_EMS.append(ut.eval_selected_metirc_storage())
        list_ES.append(ut.EarlyStopping(delta=0, patience=st.early_stopping_patience, verbose=True))


    print('training')
    """ epoch """
    ut.model_freeze(model_2, requires_grad=False)
    num_data = len(train_loader.dataset)
    for epoch in range(config.num_epochs):
        epoch = epoch + 1 # increase the # of the epoch
        print(" ")
        print("---------------  epoch {} ----------------".format(epoch))
        torch.cuda.empty_cache()

        """ print learning rate """
        for param_group in optimizer.param_groups:
            print('current LR : {}'.format(param_group['lr']))

        """ batch """
        for i, data_batch in enumerate(train_loader):
            # start = time.time()
            model_1.train()
            model_2.eval()
            EMS.total_train_step += 1

            with torch.no_grad():
                """ input"""
                datas = Variable(data_batch['data'].float()).cuda()
                # labels = Variable(data_batch['label'].long()).cuda()
                labels = Variable(data_batch['label'].float()).cuda()

                """ minmax norm"""
                if st.list_data_norm_type[st.data_norm_type_num] == 'minmax':
                    tmp_datas = datas.view(datas.size(0), -1)
                    tmp_datas -= tmp_datas.min(1, keepdim=True)[0]
                    tmp_datas /= tmp_datas.max(1, keepdim=True)[0]
                    datas = tmp_datas.view_as(datas)

                """ data augmentation """
                ##TODO : flip
                # flip_flag_list = np.random.normal(size=datas.shape[0])>0
                # datas[flip_flag_list] = datas[flip_flag_list].flip(-3)

                ##TODO : translation, cropping
                dict_result = ut.data_augmentation(datas=datas, cur_epoch=epoch)
                datas = dict_result['datas']
                # aug_dict_result = ut.data_augmentation(datas=aug_datas, cur_epoch=epoch)
                # aug_datas = aug_dict_result['datas']

                """ gaussain noise """
                # Gaussian_dist = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.01]))
                # Gaussian_dist = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), scale=torch.FloatTensor(1).uniform_(0, 0.01))
                # Gaussian_noise = Gaussian_dist.sample(datas.size()).squeeze(-1)
                # datas = datas + Gaussian_noise.cuda()

                """ model 1 forward """
                dict_result = model_2(datas)
                output_3 = dict_result['logitMap']

            """ forward propagation """
            dict_result = model_1(output_3.detach())
            output_1 = dict_result['logits']
            output_2 = dict_result['Aux_logits']
            output_3 = dict_result['logitMap']

            """ classification """
            loss_list_1 = []

            loss_2 = criterion_cls(output_1, labels)
            loss_list_1.append(loss_2)
            EMS.train_aux_loss_1.append(loss_2.data.cpu().numpy())
            loss = sum(loss_list_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """ print the train loss and tensorboard"""
            if (EMS.total_train_step) % 10 == 0 :
                # print('time : ', time.time() - start)
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      %(epoch, config.num_epochs, i + 1, (round(num_data / config.batch_size)), loss.data.cpu().numpy()))

            torch.cuda.empty_cache()
            """ pyplot """
            EMS.train_loss.append(loss.data.cpu().numpy())
            EMS.train_step.append(EMS.total_train_step)


        """ val """
        if Validation == True:
            print("------------------  val  --------------------------")
            dict_result = ut.eval_classification_model_2(config, fold, val_loader, model_1, model_2, criterion_cls)
            val_loss = dict_result['Loss']
            acc = dict_result['Acc']
            auc = dict_result['AUC']
            print('Fold : %d, Epoch [%d/%d] val Loss = %f val Acc = %f' % (fold, epoch, config.num_epochs, val_loss, acc))
            torch.cuda.empty_cache()

            """ save the metric """
            EMS.dict_val_metric['val_loss'].append(val_loss)
            EMS.dict_val_metric['val_acc'].append(acc)
            EMS.dict_val_metric['val_auc'].append(auc)
            EMS.val_step.append(EMS.total_train_step)

            """ save model """
            for i_tmp in range(len(list_selected_EMS)):
                save_flag = ut.model_save_through_validation(fold, epoch, EMS=EMS,
                                                             selected_EMS=list_selected_EMS[i_tmp],
                                                             ES=list_ES[i_tmp],
                                                             model=model_1,
                                                             dir_save_model=list_dir_save_model[i_tmp],
                                                             metric_1=st.list_standard_eval[i_tmp], metric_2='',
                                                             save_flag=False)



        if Test_flag== True:
            print("------------------  test _ test dataset  --------------------------")
            """ load data """
            dict_result = ut.eval_classification_model_2(config, fold, test_loader, model_1, model_2, criterion_cls)
            test_loss = dict_result['Loss']
            acc = dict_result['Acc']
            test_loss = dict_result['Loss']

            """ pyplot """
            EMS.test_acc.append(acc)
            EMS.test_loss.append(test_loss)
            EMS.test_step.append(EMS.total_train_step)

            print('number of test samples : {}'.format(len(test_loader.dataset)))
            print('Fold : %d, Epoch [%d/%d] test Loss = %f test Acc = %f' % (fold, epoch, config.num_epochs, test_loss, acc))
            torch.cuda.empty_cache()

        """ learning rate decay"""
        EMS.LR.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        # scheduler.step(val_loss)

        """ plot the chat """
        if epoch % 10 == 0:
            ut.plot_training_info_1(fold, dir_pyplot, EMS,  flag='percentile', flag_match=False)

        ##TODO : early stop only if all of metric has been stopped
        tmp_count = 0
        for i in range(len(list_ES)):
            if list_ES[i].early_stop == True:
                tmp_count += 1
        if tmp_count == len(list_ES):
            break

    """ release the model """
    del model_1, EMS
    torch.cuda.empty_cache()
