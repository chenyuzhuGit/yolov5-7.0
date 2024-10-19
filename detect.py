# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
注释来源：https://blog.csdn.net/qq_51511878/article/details/130004796
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle

测试命令：
    1.配置图片路径
    2.设置权重文件，为训练好的
    3.设置置信度为0.4
python detect.py --source ./VOC/images/val/000001.jpg --weights
runs/exp0/weights/best.pt --conf 0.4

NMS算法原理
NMS算法的核心思想是保留每个目标的最高置信度的边界框，同时去除其他与之高度重叠边界框。这里的重叠通常用交并比IOU来量化。
具体步骤如下：
算法分割
排序: 首先，根据每个边界框的置信度（通常是分类概率与定位准确度的综合指标）进行降序排列。置信度最高的边界框被认为是最有可能正确检测到目标的。
选择: 从排序后的列表中选择置信度最高的边界框，标记为已选，并将其添加到最终的检测结果列表中。
计算IOU: 对于剩余的每个边界框，计算它与已选边界框的IOU。
比较与剔除: 如果某个边界框与已选框的IOU超过了预设的阈值（例如0.5或0.7），则认为这两个框表示的是同一个目标，于是根据置信度较低的原则，剔除这个低置信度的边界框。
重复步骤2-4: 继续选择剩余边界框中置信度最高的，重复计算IOU和剔除过程，直到所有边界框都被检查过。
结束: 最终，剩下的边界框集合即为经过NMS处理后的检测结果，每个目标对应一个最优的边界框。

原文链接：https://blog.csdn.net/m0_74055982/article/details/138647169
"""

'''====================1.导入安装好的python库======================='''

import argparse  # 解析命令行参数的库
import csv
import os  # 与操作系统进行交互的文件库 包含文件路径操作与解析
import platform
import sys  # sys模块包含了与python解释器和它的环境有关的函数。
from pathlib import Path  # Path能够更加方便得对字符串路径进行处理

# pytorch 深度学习库
import torch

'''=====================2.获取当前文件的绝对路径=============================='''
# __file__指的是当前文件(即detect.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/detect.py
FILE = Path(__file__).resolve()
# YOLOv5 root directory  ROOT保存着当前项目的父目录,比如 D://yolov5
ROOT = FILE.parents[0]
# sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
if str(ROOT) not in sys.path:
    # add ROOT to PATH  把ROOT添加到运行路径上
    sys.path.append(str(ROOT))
# relative ROOT设置为相对路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
# 这个文件定义了Annotator类，可以在图像上绘制矩形框和标注信息
from ultralytics.utils.plotting import Annotator, colors, save_one_box

'''=====================3..加载自定义模块============================='''
# models.common这个文件定义了一些通用的函数和类，比如图像的处理、非极大值抑制等等。
from models.common import DetectMultiBackend
# utils.dataloaders这个文件定义了两个类，LoadImages和LoadStreams，
# 它们可以加载图像或视频帧，并对它们进行一些预处理，以便进行物体检测或识别。
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# 这个文件定义了一些常用的工具函数，比如检查文件是否存在、检查图像大小是否符合要求、打印命令行参数等等
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
# utils.torch_utils这个文件定义了一些与PyTorch有关的工具函数，比如选择设备、同步时间等等。
from utils.torch_utils import select_device, smart_inference_mode


'''===================1.载入参数======================='''
# smart_inference_mode：用于自动切换模型的推理模式，如果是FP16模型，则自动切换为FP16推理模式
#    否则切换为FP32推理模式，这样可以避免模型推理时出现类型不匹配的错误
@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """
    主函数
    :return:
    """
    # ---------------------初始化参数---------------------------------------------------------------------------
    # 将source转换为字符串， source为输入的图片，视频，摄像头等
    source = str(source)

    # 判断是否保存图片，如果nosave为False，且source不是txt文件，则保存图片
    save_img = not nosave and not source.endswith(".txt")  # save inference images

    # 判断source是不是视频/图像文件路径
    # Path()提取文件名。suffix：最后一个组件的文件扩展名。若source是"D://YOLOv5/data/1.jpg"， 则Path(source).suffix是".jpg"， Path(source).suffix[1:]是"jpg"
    # 而IMG_FORMATS 和 VID_FORMATS两个变量保存的是所有的视频和图片的格式后缀。
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    # 判断source是否是url，如果是，则is_url为True.lower()将字符串转换为小写,startswith()判断字符串是否以指定的字符串开头
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))

    # 判断source是否是数字，source.endswith('.streams')判断source是否以.streams结尾，
    # (is_url and not is_file)判断source是否是url，且不是文件，上述三个条件有一个为True，则webcam为True。
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)

    # 判断source是否是截图，如果是，则screenshot为True
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        # 确保输入源为本地文件，如果是url，则下载到本地，check_file()函数用于下载url文件
        source = check_file(source)  # download

    # Directories
    '''========================3.保存结果======================'''
    # 创建保存结果的文件夹
    # 增加文件或目录路径，即运行/exp——>运行/exp{sep}2，运行/exp{sep}3，…等。exist_ok为True时，如果文件夹已存在，则不会报错
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 创建文件夹，如果save_txt为True，则创建labels文件夹，否则创建save_dir文件夹
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    '''=======================4.加载模型=========================='''
    # 选择设备，如果device为空，则自动选择设备
    device = select_device(device)
    # 加载模型，DetectMultiBackend()函数用于加载模型，weights为模型路径，device为设备，dnn为是否使用opencv dnn，data为数据集，fp16为是否使用fp16推理
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    '''
        stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标
        names：保存推理结果名的列表，比如默认模型的值是['person', 'bicycle', 'car', ...] 
        pt: 加载的是否是pytorch模型（也就是pt格式的文件）
        jit：当某段代码即将第一次被执行时进行编译，因而叫“即时编译”
        onnx：利用Pytorch我们可以将model.pt转化为model.onnx格式的权重，在这里onnx充当一个后缀名称，
              model.onnx就代表ONNX格式的权重文件，这个权重文件不仅包含了权重值，也包含了神经网络的网络流动信息以及每一层网络的输入输出信息和一些其他的辅助信息。
    '''
    # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回
    stride, names, pt = model.stride, model.names, model.pt
    # 验证图像大小是每个维度的stride=32的倍数,如果不能则调整为能被整除并返回
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    '''=======================5.加载数据========================'''
    # Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    # 初始化batch_size=1
    bs = 1  # batch_size

    # 如果source是摄像头，则创建LoadStreams()对象
    if webcam:
        # 是否显示图片，如果view_img为True，则显示图片
        view_img = check_imshow(warn=True)
        # 创建LoadStreams()对象，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备，vid_stride为视频帧率
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # batch_size为数据集的长度
        bs = len(dataset)
    elif screenshot:
        # 如果source是截图，则创建LoadScreenshots()对象
        # 创建LoadScreenshots()对象，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        # 创建LoadImages()对象，直接加载图片，source为输入源，img_size为图像大小，stride为模型的stride，auto为是否自动选择设备，vid_stride为视频帧率
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # 初始化vid_path和vid_writer，vid_path为视频路径，vid_writer为视频写入对象
    vid_path, vid_writer = [None] * bs, [None] * bs

    # ---------------------开始推理---------------------------------------------------------------------------
    # Run inference：运行推理
    # warmup，预热，用于提前加载模型，加快推理速度，imgsz为图像大小，如果pt为True或者model.triton为True，则bs&#61;1，否则bs为数据集的长度。3为通道数，*imgsz为图像大小，即(1,3,640,640)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # 初始化seen，windows，dt，seen为已检测的图片数量，windows为空列表，dt为时间统计对象
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    # 遍历数据集，path为图片路径，im为图片，im0s为原始图片，vid_cap为视频读取对象，s为视频帧率
    for path, im, im0s, vid_cap, s in dataset:
        '''
         在dataset中，每次迭代的返回值是self.sources, img, img0, None, ''
          path：文件路径（即source）
          im: resize后的图片（经过了放缩操作）
          im0s: 原始图片
          vid_cap=none
          s： 图片的基本信息，比如路径，大小
        '''
        # 开始计时,读取图片
        with dt[0]:
            # # 将图片放到指定设备(如GPU)上识别。#torch.size=[3,640,480]
            # 将图片转换为tensor，并放到模型的设备上，pytorch模型的输入必须是tensor
            im = torch.from_numpy(im).to(model.device)
            # uint8 to fp16/32
            # 如果模型使用fp16推理，则将图片转换为fp16，否则转换为fp32
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 将图片归一化，将图片像素值从0-255转换为0-1
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 如果图片的维度为3，则添加batch维度
            if len(im.shape) == 3:
                # 在前面添加batch维度，即将图片的维度从3维转换为4维，即(3,640,640)转换为(1,3,640,640)，pytorch模型的输入必须是4维的
                im = im[None]  # expand for batch dim

            # 检查模型的xml属性是否为真，且第一个维度是否大于1
            if model.xml and im.shape[0] > 1:
                # 如果条件满足，会使用torch.chunk函数将im按行分割成多个张量，并将这些张量存储在名为ims的列表中
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference 推理
        # 开始计时,推理时间
        with dt[1]:
            # 可视化文件路径。如果为True则保留推理过程中的特征图，保存在runs文件夹中
            # 如果visualize为True，则创建visualize文件夹，否则为False
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 检查模型的xml属性是否为真，且第一个维度是否大于1
            if model.xml and im.shape[0] > 1:
                pred = None
                # 对分块后的图像数据进行推理，将每个图像的预测结果存储在pred中。
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                # 否则，直接对单个图像数据进行推理，将预测结果存储在pred中。
                # 推理，model()函数用于推理，im为输入图片，augment为是否使用数据增强，visualize为是否可视化,输出pred为一个列表，
                # 形状为（n,6）,n代表预测框的数量，6代表预测框的坐标和置信度，类别
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        # NMS，非极大值抑制，用于去除重复的预测框
        with dt[2]:
            # NMS，non_max_suppression()函数用于NMS，pred为输入的预测框，conf_thres为置信度阈值，iou_thres为iou阈值，classes为类别，agnostic_nms为是否使用类别无关的NMS，max_det为最大检测框数量:默认1000，
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        # 定义CSV文件的路径
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        # 创建或附加到CSV文件
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        # 处理预测结果
        # 遍历每张图片,enumerate()函数将pred转换为索引和值的形式，i为索引，det为对应的元素，即每个物体的预测框
        # 把所有的检测框画到原图中
        for i, det in enumerate(pred):  # per image
            '''
                i：每个batch的信息
                det:表示5个检测框的信息
            '''
            # 每次迭代处理一张图片
            # 检测的图片数量加1
            # seen是一个计数的功能
            seen += 1

            # 如果是摄像头，则获取视频帧率
            if webcam:  # batch_size >= 1
                # path[i]为路径列表，ims[i].copy()为将输入图像的副本存储在im0变量中，dataset.count为当前输入图像的帧数
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                # 在打印输出中添加当前处理的图像索引号i，方便调试和查看结果。在此处，如果是摄像头模式，i表示当前批次中第i张图像;否则，i始终为0，因为处理的只有一张图像。
                s += f"{i}: "
            else:
                # 如果不是摄像头，frame为0
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
                '''
                    大部分我们一般都是从LoadImages流读取本都文件中的照片或者视频 所以batch_size=1
                       p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
                       s: 输出信息 初始为 ''
                       im0: 原始图片 letterbox + pad 之前的图片
                       frame: 视频流,此次取的是第几张图片
                '''

            # 将路径转换为Path对象:图片/视频的保存路径save_path 如 runs\\detect\\exp8\\fire.jpg
            p = Path(p)  # to Path
            # im.jpg，保存图片的路径，save_dir为保存图片的文件夹，p.name为图片名称
            save_path = str(save_dir / p.name)  # im.jpg
            # im.txt，保存预测框的路径，save_dir为保存图片的文件夹，p.stem为图片名称，dataset.mode为数据集的模式，如果是image，则为图片，否则为视频
            # 设置保存框坐标的txt文件路径，每张图片对应一个框坐标信息
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            # 打印输出，im.shape[2:]为图片的宽和高
            s += "%gx%g " % im.shape[2:]  # print string
            # 得到原图的宽和高
            # 归一化因子，用于将预测框的坐标从归一化坐标转换为原始坐标
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 保存截图。如果save_crop的值为true，则将检测到的bounding_box单独保存成一张图片。
            # 如果save_crop为True，则将im0复制一份，否则为im0
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 创建Annotator对象，用于在图片上绘制预测框和标签,im0为输入图片，line_width为线宽，example为标签
            # 得到一个绘图的类，类中预先存储了原图、线条宽度、类名
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # 如果预测框的数量大于0
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测框的坐标从归一化坐标转换为原始坐标,im.shape[2:]为图片的宽和高，det[:, :4]为预测框的坐标，im0.shape为图片的宽和高
                # 将预测信息映射到原图
                # 将标注的bounding_box大小调整为和原图一致（因为训练时原图经过了放缩）此时坐标格式为xyxy
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 打印输出
                # 遍历每个类别,unique()用于获取检测结果中不同类别是数量
                for c in det[:, 5].unique():
                    # n为每个类别的预测框的数量
                    n = (det[:, 5] == c).sum()  # detections per class
                    # s为每个类别的预测框的数量和类别
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 写入结果
                # 保存预测结果：txt/图片画框/crop-image
                # 遍历每个预测框,xyxy为预测框的坐标，conf为置信度，cls为类别,reversed()函数用于将列表反转，*是一个扩展语法，*xyxy表示将xyxy中的元素分别赋值给x1,y1,x2,y2
                for *xyxy, conf, cls in reversed(det):
                    # 类别转换为int
                    c = int(cls)  # integer class
                    # label名称格式
                    label = names[c] if hide_conf else f"{names[c]}"
                    # 置信度，值类型转换
                    confidence = float(conf)
                    # 置信度字符串格式化
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        # save_csv，则将预测框的坐标和类别写入csv文件中
                        write_to_csv(p.name, label, confidence_str)

                    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id + score + xywh
                    if save_txt:  # Write to file
                        # 如果save_txt为True，则将预测框的坐标和类别写入txt文件中

                        # 将预测框的坐标从原始坐标转换为归一化坐标(可根据实际情况，是否需要做归一化处理，如：需要原始坐标时；去掉"/gn"即可)
                        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # line的形式是： ”类别 x y w h“，若save_conf为true，则line的形式是：”类别 x y w h 置信度“
                        # 如果save_conf为True，则将置信度也写入txt文件中
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # 打开txt文件,'a'表示追加
                        with open(f"{txt_path}.txt", "a") as f:
                            # 写入txt文件
                            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    # 如果save_img为True，则将预测框和标签绘制在图片上
                    # 在原图上画框+将预测到的目标剪切出来保存成图片，保存在save_dir/crops下，在原图像画图或者保存结果
                    if save_img or save_crop or view_img:  # Add bbox to image
                        # 获取类别
                        # 类别标号
                        c = int(cls)  # integer class
                        # 如果hide_labels为True，则不显示标签，否则显示标签，如果hide_conf为True，则不显示置信度，否则显示置信度
                        # 类别名
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        # 绘制预测框和标签
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    # 如果save_crop为True，则保存裁剪的图片
                    # 在原图上画框+将预测到的目标剪切出来保存成图片，保存在save_dir/crops下（单独保存）
                    if save_crop:
                        # 保存裁剪的图片
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results 在图片上绘制预测框和标签展示
            # 获取绘制预测框和标签的图片
            # 如果设置展示，则show图片 / 视频
            # im0是绘制好的图片
            im0 = annotator.result()
            # 如果view_img为True，则展示图片
            if view_img:
                # 如果系统为Linux，且p不在windows中
                if platform.system() == "Linux" and p not in windows:
                    # 将p添加到windows中
                    windows.append(p)
                    # 允许窗口调整大小,WINDOW_NORMAL表示用户可以调整窗口大小，WINDOW_KEEPRATIO表示窗口大小不变
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    # 调整窗口大小，使其与图片大小一致
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # 显示图片
                cv2.imshow(str(p), im0)
                # 等待1毫秒
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 如果save_img为True，则保存图片
            # 设置保存图片/视频
            if save_img:
                # 如果save_img为true,则保存绘制完的图片

                # 如果数据集模式为image
                if dataset.mode == "image":
                    # 保存图片
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    # 如果是视频或者"流"
                    # 如果数据集模式为video或stream
                    # 如果vid_path[i]不等于save_path
                    if vid_path[i] != save_path:  # new video
                        # vid_path[i] != save_path,说明这张图片属于一段新的视频,需要重新创建视频文件
                        # 将save_path赋值给vid_path[i]
                        vid_path[i] = save_path
                        # 以下的部分是保存视频文件
                        # 如果vid_writer[i]是cv2.VideoWriter类型
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            # 视频
                            # 获取视频的帧
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            # 获取视频的宽度
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            # 获取视频的高度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            # 视频流
                            # fps：帧数，w：宽度，h：高度
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # 获取文件保存路径
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        # 获取视频写入对象
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    # 执行视频保存到指定路径下
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # 打印时间
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    '''================7.在终端里打印出运行的结果============================'''
    # Print results 打印结果
    # 每张图片的速度,平均每张图片所耗费时间
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        # 如果save_txt为True，则打印保存的标签数量
        # 标签保存的路径
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        # 打印保存的路径
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        # 更新模型
        """
        模型训练完后，strip_optimizer函数将optimizer从ckpt中去除； 并且对模型进行model.half(), 将Float32的模型->Float16， 可以减少模型大小，提高inference速度
        """
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


'''=================三、Parse_opt()用来设置输入参数的子函数==============================='''
def parse_opt():
    """
    初始化运行参数
    :return:
    """

    # 创建参数解析对象
    parser = argparse.ArgumentParser()
    # 以下配置有default-----------------------------
    # 训练的权重路径，可以使用自己训练的权重，也可以使用官网提供的权重。默认官网的权重
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    # source 也可以是视频文件
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    # 配置数据文件路径，包括image/label/classes等信息，训练自己的文件，需要作相应更改，可以不用管
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    # 预测时网络输入图片的尺寸，默认值为 [640]
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    # 置信度，表示置信度为多少时(检测过程中，会在系统指定的类型集合中，计算出这个目标对应每个类型的置信度值)，认为这是一个目标对象，就会框出来；是一个阈值，根据业务需求设置
    # 简单理解，置信度大于这个阈值时，就认为这是一个目标对象
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    # iou置信度，实际检测时，一个目标会有被多个框框中，这时需要选中最佳位置的一个框。NMS(非极大值抑制)机制就是解决这个问题，根据框与框的IOU(两个框交集除以并集的值)那个最大，选出最合适的框。
    # iou-thres就是配置这个阈值的，先根据IOU排序，以IOU最大的作为基础对比框(IOU最大的留下的概率也最高)，IOU大于多少时，会合并两个框(去掉IOU低的那个框)，依次执行，直至都不满足阈值为止
    # 原理参考本文件的备注信息(37行处)
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    # 保留的最大检测框数量，每张图片中检测目标的个数最多为1000类
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    # 执行的设备默认为空，会自动检测。
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")

    # 以下配置没有default(只要配置这个参数(可以不写值)，这个参数就是true)-----------------------------
    # 训练完成时，是否自动显示结果
    parser.add_argument("--view-img", action="store_true", help="show results")
    # 是否把结果。保存为txt（结果exp文件夹，会增加一个labels文件夹，保存了相关内容）
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    # 是否保存检测结果的置信度到 txt文件，默认为 False
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    # 是否保存裁剪预测框图片，默认为False，使用--save-crop 在runs/detect/exp*/crop/剪切类别文件夹/ 路径下会保存每个接下来的目标
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    # 不保存图片、视频，要保存图片，不设置--nosave 在runs/detect/exp*/会出现预测的结果
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    # 配置，结果图片只展示那些类型的框，其余的不展示
    # nargs="+",表示可以配置多个值
    # 如下配置只展示"0 2 3"三个类型的框；根据需求设置
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    # nms 增强
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    # 是否使用数据增强进行推理，默认为 False
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    # b是否可视化特征图，默认为 False
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    # 减少模型中的一些组件
    parser.add_argument("--update", action="store_true", help="update all models")
    # 结果默认保存的路径，默认为'runs/detect'
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    # 结果默认保存的名称，默认为'runs/detect'路径下，exp1，exp2，exp3……
    parser.add_argument("--name", default="exp", help="save results to project/name")
    # 每次的执行结果，是否都保存在一个文件夹里，(默认每次都创建新的结果文件夹：exp,exp1,exp2……)
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 画 bounding box 时的线条宽度，默认为 3
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    # 是否隐藏标签信息，默认为 False
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    # 是否隐藏置信度信息，默认为 False
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    # 是否使用 FP16 半精度进行推理，默认为 False
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # 是否使用 OpenCV DNN 进行 ONNX 推理，默认为 False
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    # 解析命令参数，并赋值给opt;扩充维度
    opt = parser.parse_args()
    # 如果只有一个参数，则将其扩展为两个参数，对应图片的高和宽
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 打印参数，vars()函数返回对象object，的属性和属性值的字典对象
    print_args(vars(opt))
    # 返回参数
    return opt

'''=======================二、设置main函数==================================='''
def main(opt):
    """
    主函数
    :param opt:
    :return:
    """

    # 检查环境/打印参数,主要是requrement.txt的包是否安装，用彩色显示设置的参数
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    # 运行程序，vars()函数返回对象object的属性和属性值字典对象
    run(**vars(opt))


# 命令使用
# python detect.py --weights best.pt --source  data/images/fishman.jpg # webcam
if __name__ == "__main__":
    opt = parse_opt() # 解析参数
    main(opt) # 执行主函数
