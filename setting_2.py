""" flags """
""" preprocessing for the data """
flag_orig_npy = False
flag_fold_index = False
flag_orig_npy_other_dataset = False
flag_print_trainAcc = False

""" flag whether to use pre-train model"""
flag_classification = False
flag_classification_using_pretrained = False
flag_classification_fine_tune = False
flag_bayesian = False
flag_regression = False
flag_multi_task = True

""" loss """
flag_loss_1 = True
flag_loss_2 = True
flag_loss_3 = False
flag_loss_4 = False

# val
flag_loss_5 = False

""" training strategy"""
flag_estimate_age = False
flag_ttest = False
flag_downSample = False
flag_gaussian_norm = False
flag_RoI_template = False

"""
0 : always      
1 : no cropping       
2 : (1:1) cropping
"""
flag_MC_dropout = False
flag_translation = True
flag_translation_subject_wise = True
flag_translation_ratio = 0
flag_translation_ratio_2 = False
flag_eval_translation = False

flag_cropping = False
flag_eval_cropping = False
flag_crop_ratio = 0

flag_Gaussian_blur = False
