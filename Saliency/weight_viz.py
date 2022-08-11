from model_2 import SalGAN_3, LOSS,SalGAN_4,SalGAN_2
import torch
import os 
import matplotlib.pyplot as plt
from torch import nn

net =  SalGAN_3()
net1 =  SalGAN_3()
model = SalGAN_4()
model1 = SalGAN_4()

weight = torch.load("./models_norm/VGG/ckpt_epoch_250_0.7.pth")
net1.encoder_salgan.load_state_dict(weight['model'], strict=False)  

res_weight  = torch.load("./models_norm/RESNET/resnet_selfattended.pth")
model_dict =  model.state_dict()
pretrained_dict = {"encoder_salgan."+'.'.join(k.split('encoder1')[-1].split('.')[1:]):
                    v for k, v in res_weight['model'].items() if 'encoder1' in k.split('.')}
model_dict.update(pretrained_dict)
model.load_state_dict(pretrained_dict, strict=False)

temp_w ={}
cout =0

for j, m in enumerate(net.encoder_salgan.modules()):
    if isinstance(m, nn.Conv2d):
        print(cout , j)
        temp_w[cout] = m.weight.data      
        cout+=1


del res_weight

# with torch.no_grad():
#     for k, m in enumerate(net.encoder.module.encoder_salgan.modules()):
#         if isinstance(m, nn.Conv2d):
#             # print(i, m.weight.data.size())
#             #copy salgan encoder weights
#             temp_w[i] = m.weight.data

model_paths = './models_norm/VGG/'
# model_paths = /home/tarun/Documents/Phd/360_cvpr/models/memory_nce_16000_lr_0.01_decay_0.0001_bsz_25_optim_SGD
app = 'ckpt_epoch_1.pth'


"""
    =====================================
"""


path = os.path.join(model_paths, app)
checkpoint = torch.load(path, map_location='cpu')
net.load_state_dict(checkpoint['model'], strict=False)


res_weight  = torch.load("./models_norm/RESNET/ckpt_epoch_1.pth")
model_dict1 =  model1.state_dict()
pretrained_dict = {"encoder_salgan."+'.'.join(k.split('encoder1')[-1].split('.')[1:]):
                    v for k, v in res_weight['model'].items() if 'encoder1' in k.split('.')}
model_dict1.update(pretrained_dict)
model1.load_state_dict(pretrained_dict, strict=False)



max_w = []
mean_w =[]
x_axis =[]
cout=0
import numpy as np

for j, m in enumerate(net1.encoder_salgan.modules()):

    if isinstance(m, nn.Conv2d):
          
        if(temp_w[cout].size() == m.weight.data.size()):
            # mean_w.append((temp_w[cout]- m.weight.data).abs().max())
            # print(m.weight.data.size(), m.weight.data.size()[0],temp_w[cout].size()[1])
            max_w.append((torch.norm(temp_w[cout] - m.weight.data))/torch.norm(temp_w[cout]))  #/(m.weight.data.cpu().numpy().size)
            # mean_w.append(torch.norm(temp_w[cout]))
            # max_w.append((torch.norm(m.weight.data)))   
              
            x_axis.append(str(cout))
            cout+=1

# plt.plot(x_axis, mean_w,'ro')
# plt.xlabel('Conv layers')
# plt.ylabel('Mean weight diff')
# plt.show()
# plt.close()
plt.plot(x_axis, max_w,'ro')
plt.xlabel('VGG Conv layers')
plt.ylabel('Frobenius Norm weight difference')
plt.grid()
plt.show()
plt.close()


"""
    =====================================
"""



















# diff ={}
# # print(os.listdir(model_paths))


# for i in range(1, 201):
#     print(app+str(i)+'.pth')
#     path = os.path.join(model_paths, app+str(i)+'.pth')
#     try:
#         diff[app+str(i)+'.pth'] = _
        
#     except:
#         # print('hell')
#         diff[app+str(i)+'.pth']=[]
#         if os.path.isfile(path):
#             checkpoint = torch.load(path, map_location='cpu')
#             net.load_state_dict(checkpoint['model'])
#             for j, m in enumerate(net.encoder.module.encoder_salgan.modules()):
#                 if isinstance(m, nn.Conv2d):
#                     if(temp_w[j].size() == m.weight.data.size()):
#                         # print(j, m.weight.data.size())
#                         # diff = torch.norm(temp_w[j]-m.weight.data).item()
#                         # print(torch.norm(temp_w[j]-m.weight.data).item())
#                         diff[app+str(i)+'.pth'].append(torch.norm(temp_w[j]-m.weight.data).item())
#                         # temp_w[j] = m.weight.data
#             del checkpoint

#     plt.plot(range(len(diff[app+str(i)+'.pth'])) ,diff[app+str(i)+'.pth'],'ro') 
#     plt.show()
#     plt.close()
    