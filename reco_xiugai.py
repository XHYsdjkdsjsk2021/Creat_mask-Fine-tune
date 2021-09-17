import cv2 
import numpy as np
import os
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import time
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm,trange
import pickle
from pprint import pprint



def seam(final_time_right_down):
    final = []
    sum = []
    condition = False
    if len(final_time_right_down) <= 1:
        final = final_time_right_down
    else:
        for iii in range(len(final_time_right_down)-1):
            if final_time_right_down[iii][1] - final_time_right_down[iii+1][0] < 50:         
                change = final_time_right_down[iii]+final_time_right_down[iii+1]
                sum = sum + change  
                condition = True
            if final_time_right_down[iii][1] - final_time_right_down[iii+1][0] >= 50 or iii == len(final_time_right_down)-2:
                if condition is True:
                    change_ = []
                    change_.append(sum[0])
                    change_.append(sum[len(sum)-1])
                    final.append(change_)
                    sum = []
                    condition = False
                else:
                    stay = final_time_right_down[iii]
                    final.append(stay)
                    sum = []
                    condition = False
    return final

def get_time(right_down):
    final_time_right_down = []
    for fr in range(len(right_down)-1):
        frame_pre = right_down[fr]
        frame_after = right_down[fr+1]
        if fr==0 and right_down[0] == 'None.jpg':
            start = 0
            end = 0
        if fr==0 and right_down[0] != 'None.jpg':
            start = 1
            end = 0
        if frame_pre == 'None.jpg' and frame_after == 'None.jpg':
            pass
        if (frame_pre != 'None.jpg' and frame_after != 'None.jpg'):
            end = (fr+1)*10+1+10
        if (frame_pre == 'None.jpg' and frame_after != 'None.jpg'):
            start = (fr+1)*10+1
        if (frame_pre != 'None.jpg' and frame_after == 'None.jpg'):
            end = (fr)*10+1
            res_real = []
            res_real.append(start)
            res_real.append(end)
            final_time_right_down.append(res_real)
        if fr == len(right_down)-2 and right_down[len(right_down)-1] != 'None.jpg':
            res_real = []
            res_real.append(start)
            res_real.append(end)
            final_time_right_down.append(res_real)
    return final_time_right_down

class FeatureExtractor(nn.Module): # 提取特征工具
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": 
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


# model = models.resnet50(pretrained=True) # 加载resnet50工具
# model = model.cuda()
# model.eval()

model = models.resnet50()
channel_in = model.fc.in_features
num_classes = 9
model.fc = nn.Linear(channel_in, num_classes)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('/data/haoyuan/model_50.pth'))
model = model.cuda()
model.eval()

def input_image(img_standard):
    img_standard=cv2.resize(img_standard,(224,224))
    img_standard=cv2.cvtColor(img_standard,cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_standard=transform(img_standard).cuda()
    img_standard=img_standard.unsqueeze(0)
    # model2 = FeatureExtractor(model, ['layer4']) # 指定提取 layer3 层特征
    with torch.no_grad():
        out_=model(img_standard)
    return out_


def get_location(mask,num):
        if num <50:
            return 0,0,0,0
        else:
            top = np.min(np.where(mask)[0]) 
            left = np.min(np.where(mask)[1])
            down = np.max(np.where(mask)[0])
            right = np.max(np.where(mask)[1])
        return top ,left,down,right

def detect(img1,img2,height_cor,width_cor):
    # img1 = cv2.medianBlur(img1, 5)
    gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    edge_1 = cv2.Canny(gray_1, 150, 300) #(200,300)
    edge_2 = cv2.Canny(gray_2, 150, 300)
    small_height = int(height*height_cor)
    small_width = int(width*width_cor)
    mask = np.zeros([small_height, small_width], dtype=np.uint8)
    return mask,small_height,small_width,edge_1,edge_2,img1

#当standard库中元素更新时 打开这个更新pkl
filepath='/data/haoyuan/Standard/'
filenames = os.listdir(filepath)
filenames.sort(key=lambda x: int(x[:-4]))
fil = open('standard.pkl', 'wb')
for filename in filenames:
    img_standard = cv2.imread(filepath+filename)
    out_ = input_image(img_standard)
    out_ = out_.cpu().numpy()
    pickle.dump(out_, fil,1)
fil.close()

right_down=[]
left_down =[]
left_up=[]
right_up = []
img_path ='/data/haoyuan/audit_data/logo_test/2021_08_31_10_18_10/' #抽帧图片路径，需要维护
file_number = 26 #Standard库中存在的标个数 需要维护
frame = len(os.listdir(img_path))

start = time.time()
for nnn in range(int(frame/10)):  
    tem = 10*(nnn)+1
    img1 = cv2.imread(img_path+'%05d' %(1*tem)+'.jpg')
    img2 = cv2.imread(img_path+'%05d' %(1*tem+10)+'.jpg')
    height = img1.shape[0] #1080
    width = img1.shape[1]  #1920
    
    # 右上
    img1_right = img1[0:int(height*0.2),int(width*0.8):width]
    img2_right = img2[0:int(height*0.2),int(width*0.8):width]

    #左上
    img1_left = img1[0:int(height*0.2),0:int(width*0.2)]
    img2_left = img2[0:int(height*0.2),0:int(width*0.2)]

    #左下
    img1_left_down = img1[int(height*0.6):height,0:int(width*0.2)]
    img2_left_down = img2[int(height*0.6):height,0:int(width*0.2)]

    #右下
    img1_right_down = img1[int(height*0.7):height,int(width*0.8):width]
    img2_right_down = img2[int(height*0.7):height,int(width*0.8):width]


    mask,small_height,small_width,edge_1,edge_2,img_or = detect(img1_right,img2_right,0.2,0.2)
    edge_1 = edge_1.astype('uint8')
    edge_2 = edge_2.astype('uint8')
    mask = (edge_1+edge_2)==254
    num = np.sum(mask)
    top ,left,down,right = get_location(mask,num)   
    if num < 50 or top == down or left == right:
        final = 'None.jpg'
        right_up.append(final)
    else:
        img_or = img_or[top:down,left:right]
        kk_ = 0
        out = input_image(img_or)
        out = out.cpu().numpy()
        fie = open('standard.pkl', 'rb')
        for oo in range(file_number):
            cos = cosine_similarity(out,out_)
            b = np.sum(np.sum(cos))
            re = b /(cos.shape[0]*cos.shape[1])
            if re > kk_:
                kk_ = re
                final = '第'+str(oo)+'类'
        right_up.append(final)
    
    mask_left,small_height_left,small_width_left,edge_1_left,edge_2_left,or_left = detect(img1_left,img2_left,0.2,0.2)
    edge_1 = edge_1_left.astype('uint8')
    edge_2 = edge_2_left.astype('uint8')
    mask_left = (edge_1+edge_2)==254
    num = np.sum(mask_left)
    top ,left,down,right = get_location(mask_left,num)
    if num <50 or top == down or left == right:
        final = 'None.jpg'
        left_up.append(final)
    else:
        or_left = or_left[top:down,left:right]
        kk_ = 0
        out = input_image(or_left)
        # out = input_image_Surf(or_left)
        out = out.cpu().numpy()
        fie = open('standard.pkl', 'rb')
        for oo in range(file_number):
            out_ = pickle.load(fie)
            cos = cosine_similarity(out,out_)
            b = np.sum(np.sum(cos))
            re = b /(cos.shape[0]*cos.shape[1])
            if re > kk_:
                kk_ = re
                final = '第'+str(oo)+'类'
        left_up.append(final)


    mask_left_down,small_height_left_down,small_width_left_down,edge_1_left_down,edge_2_left_down,or_left_down = detect(img1_left_down,img2_left_down,0.4,0.2)
    edge_1 = edge_1_left_down.astype('uint8')
    edge_2 = edge_2_left_down.astype('uint8')
    mask_left_down = (edge_1+edge_2)==254
    num = np.sum(mask_left_down)
    top ,left,down,right = get_location(mask_left_down,num)
    if num < 50 or top == down or left == right:
        final = 'None.jpg'
        left_down.append(final)
    else:
        or_left_down = or_left_down[top:down,left:right]
        kk_ = 0
        out = input_image(or_left_down)
        out = out.cpu().numpy()
        fie = open('standard.pkl', 'rb')
        for oo in range(file_number):
            out_ = pickle.load(fie)
            cos = cosine_similarity(out,out_)
            b = np.sum(np.sum(cos))
            re = b /(cos.shape[0]*cos.shape[1])
            if re > kk_:
                kk_ = re
                final = '第'+str(oo)+'类'
        left_down.append(final)

    mask_right_down,small_height_right_down,small_width_right_down,edge_1_right_down,edge_2_right_down,or_right_down = detect(img1_right_down,img2_right_down,0.3,0.2)
    edge_1 = edge_1_right_down.astype('uint8')
    edge_2 = edge_2_right_down.astype('uint8')
    mask_right_down = (edge_1+edge_2)==254
    num = np.sum(mask_right_down)
    top ,left,down,right = get_location(mask_right_down,num)
    if num < 50 or top == down or left == right:
        final = 'None.jpg'
        right_down.append(final)
    else:
        or_right_down = or_right_down[top:down,left:right]
        kk_ = 0
        out = input_image(or_right_down)
        out = out.cpu().numpy()
        fie = open('standard.pkl', 'rb')
        for oo in range(file_number):
            out_ = pickle.load(fie)
            cos = cosine_similarity(out,out_)
            b = np.sum(np.sum(cos))
            re = b /(cos.shape[0]*cos.shape[1])
            if re > kk_:
                kk_ = re
                final = '第'+str(oo)+'类'
        print(final)
        print(kk_)
        print(img_path+'%05d' %(1*tem)+'.jpg')
        right_down.append(final)

right_down_data = np.unique(right_down)
right_up_data = np.unique(right_up)
left_down_data = np.unique(left_down)
left_up_data = np.unique(left_up)

final_time_right_down = get_time(right_down)
final_time_right_up = get_time(right_up)
final_time_left_down = get_time(left_down)
final_time_left_up = get_time(left_up)

final_time_right_down = seam(final_time_right_down)
final_time_right_up = seam(final_time_right_up)
final_time_left_down = seam(final_time_left_down)
final_time_left_up = seam(final_time_left_up)


resdata = []
for ii in right_down_data:
    resdata.append(right_down.count(ii))
for ixv in range(len(resdata)):
    if resdata[ixv]==max(resdata):
        final_result = right_down_data[ixv]
print('rightdown=',final_result)
if final_result == 'None.jpg':
    print('rightdown_time=no time')
else:
    print('rightdown_time=',final_time_right_down)

resdata_1 = []
for ii in right_up_data:
    resdata_1.append(right_up.count(ii))
for ixv in range(len(resdata_1)):
    if resdata_1[ixv]==max(resdata_1):
        final_result = right_up_data[ixv]
print('rightup=',final_result)
if final_result == 'None.jpg':
    print('rightup_time=no time')
else:
    print('rightup_time=',final_time_right_up)

resdata_2 = []
for ii in left_up_data:
    resdata_2.append(left_up.count(ii))
for ixv in range(len(resdata_2)):
    if resdata_2[ixv]==max(resdata_2):
        final_result = left_up_data[ixv]

print('leftup=',final_result)
if final_result == 'None.jpg':
    print('leftup_time=no time')
else:
    print('leftup_time=',final_time_left_up)

resdata_3 = []
for ii in left_down_data:
    resdata_3.append(left_down.count(ii))
for ixv in range(len(resdata_3)):
    if resdata_3[ixv]==max(resdata_3):
        final_result = left_down_data[ixv]

print('leftdown=',final_result)
if final_result == 'None.jpg':
    print('leftdown_time=no time')
else:
    print('leftdown_time=',final_time_left_down)
            
end = time.time()
print("总耗时=",end -start,'s')    



