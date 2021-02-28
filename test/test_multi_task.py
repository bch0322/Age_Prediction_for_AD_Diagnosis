import nibabel as nib
import numpy as np
import setting as st
import setting_2 as fst
from data_load import data_load as DL
import torch
from torch.autograd import Variable
import torch.nn as nn
import utils as ut
import os
from scipy import stats
from sklearn.metrics import confusion_matrix

def test(config, fold, model, loader, dir_to_load,  dir_confusion):
    """ free all GPU memory """
    torch.cuda.empty_cache()
    # criterion_cls = nn.CrossEntropyLoss()
    # criterion_cls = ut.FocalLoss(gamma=st.focal_gamma, alpha=st.focal_alpha)
    criterion_cls = nn.BCELoss()
    criterion_L1 = nn.L1Loss(reduction='sum').cuda()
    test_loader = loader

    """ load the model """
    model_dir = ut.model_dir_to_load(fold, dir_to_load)
    if model_dir !=None:
        model.load_state_dict(torch.load(model_dir))
    model.eval()

    dict_result = ut.eval_multi_task_model(config, fold, test_loader, model, criterion_cls, criterion_L1)


    return dict_result
