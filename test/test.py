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
    test_loader = loader

    """ load the model """
    model_dir = ut.model_dir_to_load(fold, dir_to_load)
    if model_dir !=None:
        model.load_state_dict(torch.load(model_dir))
    model.eval()

    if fst.flag_eval_cropping == True:
        dict_result = ut.eval_classification_model_cropped_input(config,fold, test_loader, model, criterion_cls)
    elif fst.flag_eval_translation == True:
        dict_result = ut.eval_classification_model_esemble(config, fold, test_loader, model, criterion_cls)
    elif fst.flag_MC_dropout == True:
        dict_result = ut.eval_classification_model_MC_dropout(config, fold,  test_loader, model, criterion_cls)
    elif fst.flag_bayesian == True:
        dict_result = ut.eval_classification_model_bayesian(config, fold, test_loader, model, criterion_cls)
    else:
        dict_result = ut.eval_classification_model(config, fold,  test_loader, model, criterion_cls, flag_heatmap=False)


    return dict_result
