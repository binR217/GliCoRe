
import pickle
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

#源plans文件的路径，基于该文件进行修改,这个保存batchsize的文件在processed文件夹对应的任务id下，请根据实际情况修改下面路径
path = '<PATH_TO_TASK003_OUTPUT>/glicore_Plansv2.1_plans_3D.pkl'
f = open(path, 'rb')
plans = pickle.load(f)
#可以通过print命令查看整个文件的内容，格式为类似json的结构，可以用一些json格式化工具查看，具体度娘
# print(plans)

# print("--------------分割线--------------")
# 查看原来的patchsize
print(plans['plans_per_stage'][0]['patch_size'])
print(plans['plans_per_stage'][0]['batch_size'])
# print(plans['plans_per_stage'][1]['patch_size'])


plans = load_pickle(path)

# # 例如，plans 更改patchsize 将batchsize改为6 patchsize改为48*192*192
plans['plans_per_stage'][0]['batch_size'] = 2
# plans['plans_per_stage'][0]['patch_size'] = np.array((48, 192, 192))
#
# plans['plans_per_stage'][1]['batch_size'] = 6
# plans['plans_per_stage'][1]['patch_size'] = np.array((48, 192, 192))

# save the plans under a new plans name. Note that the new plans file must end with _plans_2D.pkl!
#保存到默认的路径下，这样才能被识别，必须以_plans_2D.pkl或者_plans_3D.pkl结尾；可以按照以下方式命名方便通过文件名识别batchsize的大小
save_pickle(plans, join("<PATH_TO_TASK003_OUTPUT>/",
                        'glicore_Plansv2.1_plans_3D.pkl'))
print(plans['plans_per_stage'][0]['patch_size'])
print(plans['plans_per_stage'][0]['batch_size'])
