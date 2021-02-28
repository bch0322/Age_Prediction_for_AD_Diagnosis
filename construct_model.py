import setting as st

""" model """
from model_arch import resnet
from model_arch import resnet_age

def construct_model(config, flag_model_num = 0):
    """ construct model """
    if flag_model_num == 0:
        model_num = st.model_num_0
    elif flag_model_num == 1:
        model_num = st.model_num_1
    elif flag_model_num == 2:
        model_num = st.model_num_2


    if model_num == 0:
        model = resnet.resnet18().cuda()
    elif model_num == 1:
        model = resnet.resnet50().cuda()
    elif model_num == 5:
        model = resnet_age.resnet18().cuda()

    return model

