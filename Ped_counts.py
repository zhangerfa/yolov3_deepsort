# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:02:05 2021

@author: erfa_zhang
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.utils import convert_boxes
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
from yolov3_tf2.dataset import transform_images

'''
def signLine(SXNum,bmxNum,imgName):
    # 输入实线个数，斑马线个数，通过点击返回实线和斑马线坐标
    # 读入第一帧
    sx = [] # 实线
    bmx = [] # 斑马线
    im = np.array(Image.open(imgName)) # 第一帧像素矩阵
    plt.ion() # 打开交互模式
    # 展示第一帧并标记出实线
    plt.imshow(im)
    for i in range(SXNum):
        print('Please click 2 points')
        x = plt.ginput(2)
        print('you clicked:',x)
        sx.append(x)
        print("已标记一条车道分界线")
    # 标记出斑马线 
    for i in range(bmxNum):
        print('Please click 4 points')
        x = plt.ginput(4)
        print('you clicked:',x)
        bmx.append(x)
        print("已标记一条斑马线")
    # 交互结束
    plt.ioff()    
    return sx,bmx

def linePara(sx,bmx):
    # 输入标记点坐标，输出绘图所用参数
    
    # 获取实线参数（斜率，截距）
    def shixian(x1,y1,x2,y2):
        if x1 == x2: # 垂线
            k = -999
            b = 0
        else:
            k = (y2-y1)/(x2-x1) # 斜率
            b = y1 - x1 * k # 截距
        return k,b
    
    sxPara = [] # 实线参数列表
    # 计算实线的值 y = kx + b
    for i in sx:
        # 遍历变量是两个元组（点坐标）组成的列表
        y1 = int(i[0][1])
        y2 = int(i[1][1])
        if y1 > y2:
            # 保证y1<y2
            x1 = int(i[1][0])
            y1 = int(i[1][1])
            x2 = int(i[0][0])
            y2 = int(i[0][1])
        else:
            x1 = int(i[0][0])
            x2 = int(i[1][0])
        # 获取该实线的斜率，截距
        sxPara.append([shixian(x1,y1,x2,y2),x1,x2,y1,y2])
        
        bmxPara = [] #斑马线参数列表
        for i in bmx:
            # bmx是列表，每个元素是四个元组（点坐标）
            x1 = int(i[0][0])
            y1 = int(i[0][1])
            x2 = int(i[1][0])
            y2 = int(i[1][1])
            x3 = int(i[2][0])
            y3 = int(i[2][1])
            x4 = int(i[3][0])
            y4 = int(i[3][1])
            # cv2.rectangle 只能画正矩形
            xmax = max(x1,x2,x3,x4)
            xmin = min(x1,x2,x3,x4)
            ymax = max(y1,y2,y3,y4)
            ymin = min(y1,y2,y3,y4)
            bmxPara.append([xmax,xmin,ymax,ymin])
    return sxPara,bmxPara

def drawLine(sxPara,bmxPara,img):
    # 输入被标记图片，实线，斑马线参数，绘制实线，斑马线

    for i in sxPara:
        k = i[0][0]
        b = i[0][1]
        x1 = i[1]
        x2 = i[2]
        y1 = i[3]
        y2 = i[4]
        # 绘制实线
        lineLidth = 5 #道路分界线宽度
        if k != 0:
            for xxx in range(y1,y2):
                # 遍历y1~y2之间的纵坐标
                xq = (xxx-b)/k
                xq = int(xq) # 计算出对应横坐标
                cv2.rectangle(img, (xq+lineLidth,xxx), (xq-lineLidth,xxx), (0,0,255), -1)
        else:
            for xxx in range(x1,x2):
                # 水平线遍历横坐标
                yq = b #纵坐标等于常数截距
                cv2.rectangle(img, (xxx,yq+lineLidth), (xxx,yq-lineLidth), (0,0,255), -1)
    
    # 绘制斑马线
    for i in bmxPara:
        xmax = i[0]
        xmin = i[1]
        ymax = i[2]
        ymin = i[3]
        # 汇出斑马线
        cv2.rectangle(img, (xmax,ymax), (xmin,ymin), (0,255,0), 4)
'''   
def initTracker():
    # 初始化追踪器：初始化追踪器并载入权重信息和分类信息    
    # 追踪器参数
    max_cosine_distance = 0.5
    nn_budget = None
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # 初始化追踪器
    tracker = Tracker(metric)
    yolo = YoloV3(classes=80)
    # 载入权重信息
    weightsPath = r'.\weights\yolov3.tf'
    yolo.load_weights(weightsPath)
    print("载入权重信息成功")
    # 载入分类信息
    class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
    print('载入分类信息成功')
    return yolo,class_names,tracker
    
        
def Ped_counts(img,yolo,class_names,tracker,car_num,ped_num):
    # 行人追踪
    
    nms_max_overlap = 1.0
    model_filename = 'model_data/mars-small128.pb' # 使用已训练好的模型
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # 颜色通道转换
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_in = tf.expand_dims(img_in, 0)
    # 输入图像图形尺寸要调整，否则报错
    img_in = transform_images(img_in, 416)
    
    #### 行人检测
    boxes, scores, classes, nums = yolo.predict(img_in,steps = 1)
    classes = classes[0]
    print(classes)
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])]) # 在分类信息中获取物体分类
    names = np.array(names)
    # 检测目标边框
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)
    # 获取检测得到的目标
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
    print(type(detections))
    print(detections)
    #initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    # run non-maxima suppresion
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]        

    #### 行人追踪
    tracker.predict()
    tracker.update(detections)
    print(len(tracker.tracks))
    
    #### 行人计数
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 

        bbox = track.to_tlbr()
        class_name = track.get_class()
        if class_name == 'car':
            car_num += 1
        elif class_name == 'person':
            ped_num += 1 
        # 为不同追踪器分配不同颜色
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        # 目标标记
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        # 展示名称
        cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]) , int(bbox[1]-10)),0, 0.75, (255,255,255),2) 
    return converted_boxes

def main(video,SXNum,bmxNum):
    #输入视频路径，车道分界线，斑马线个数开始程序
    
    # 读入第一帧并保存
    imgName = "firstImg.png" #第一帧储存名
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    n=1
    while n < 10: #1000帧之后出现场景
    	success, image = vidcap.read()
    	n+=1
    cv2.imwrite(imgName,image)
    print("第一帧读入成功并保存为"+imgName)
    '''
    # 用户标记车道分割线和斑马线
    sx,bmx = signLine(SXNum,bmxNum,imgName)
    # 获取标记线段参数
    sxPara,bmxPara = linePara(sx, bmx)
    plt.close()
    '''
    # 输出视频
    output = r'..\video\output.avi'
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, codec, fps, (width, height))
    list_file = open('detection.txt', 'w')
    fps = 0.0

    # 初始化追踪器
    yolo,class_names,tracker = initTracker()
    
    car_num = 0 # 车辆计数器
    ped_num = 0 # 行人计数器
    # 遍历视频
    while True:
        ok,img = vidcap.read()
        # 读取完毕退出
        if not ok:
            break
        # 行人计数
        converted_boxes = Ped_counts(img,yolo,class_names,tracker,car_num,ped_num)
        # 汇出道路分界线和实线
        #drawLine(sxPara,bmxPara,img)
        # 展示
        cv2.putText(img, 'carNum:'+str(car_num)+'    '+'PedNum:'+str(ped_num), (0, 30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2    )
        img = cv2.resize(img,(0,0), fx=1.5,fy=1.5)
        #cv2.imshow("Ped_counts",img)
        
        # 储存该帧
        out.write(img)
        '''
        frame_index = frame_index + 1
        list_file.write(str(frame_index)+' ')
        if len(converted_boxes) != 0:
            for i in range(0,len(converted_boxes)):
                list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
        list_file.write('\n')
        '''
        #if cv2.waitKey(1) & 0xFF == 27: #esc推出
        # 一帧处理完等待waitkey(毫米)再处理下一帧
            #break
        
    
    # 关闭所有窗口
    #vidcap.release()
    #out.release()
    #list_file.close()
    #cv2.destroyAllWindows()


#### 运行主程序
video = '../video/test.mp4'
SXNum = 1 # 道路分界线个数
bmxNum = 1 # 斑马线个数
main(video,SXNum,bmxNum) 


