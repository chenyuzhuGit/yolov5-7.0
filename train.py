# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
注释来源：https://blog.csdn.net/yrhzmu/article/details/135192283?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-135192283-blog-135184220.235^v43^pc_blog_bottom_relevance_base7&spm=1001.2101.3001.4242.1&utm_relevant_index=1
阿里云讲解（优）:https://developer.aliyun.com/article/1309712
train.py和val.py和detect.py三个文件的关系：https://blog.csdn.net/qq_53092944/article/details/136857783
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.
在数据集上训练 yolo v5 模型

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    训练数据为coco128 coco128数据集中有128张图片 80个类别，是规模较小的数据集
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data

测试：
python train.py --data data/VOC-new.yaml --cfg models/yolov5s-voc.yaml --weights  weights/yolov5s.pt --batch-size 8 --workers 2 --epochs 20
"""

'''======================1.导入安装好的python库====================='''
import argparse  # 解析命令行参数模块
import math  # 数学公式模块
import os  # 与操作系统进行交互的模块 包含文件路径操作和解析
import random  # 生成随机数模块
import subprocess  # 创建子进程
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
import time  # 时间模块 更底层
from copy import deepcopy  # 深度拷贝模块
from datetime import datetime, timedelta  # datetime模块能以更方便的格式显示日期或对日期进行运算。
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块

try:
    # 用于跟踪、比较、解释和优化机器学习模型和实验的平台
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np  # numpy数组操作模块
import torch  # 引入torch
import torch.distributed as dist  # 分布式训练模块
import torch.nn as nn  # 对torch.nn.functional的类的封装 有很多和torch.nn.functional相同的函数
import yaml  # yaml是一种直观的能够被电脑识别的的数据序列化格式，容易被人类阅读，并且容易和脚本语言交互。一般用于存储配置文件。
from torch.optim import lr_scheduler  # tensorboard模块
from tqdm import tqdm  # 进度条模块

'''===================2.获取当前文件的绝对路径========================'''
# __file__指的是当前文件(即train.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/train.py
FILE = Path(__file__).resolve()
# ROOT保存着当前项目的父目录,比如 D://yolov5
ROOT = FILE.parents[0]  # YOLOv5 root directory
# sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
if str(ROOT) not in sys.path:
    # add ROOT to PATH  把ROOT添加到运行路径上
    sys.path.append(str(ROOT))
# relative ROOT设置为相对路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

'''===================3..加载自定义模块============================'''
import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import LOGGERS, Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)

'''================4.分布式训练初始化==========================='''
# https://pytorch.org/docs/stable/elastic/run.html该网址有详细介绍
# rank & local_rank:用于表示进程的序号，用于进程间通信。每一个进程对应了一个rank。
'''
   查找名为LOCAL_RANK，RANK，WORLD_SIZE的环境变量，
   若存在则返回环境变量的值，若不存在则返回第二个参数（-1，默认None）
rank和local_rank的区别： 两者的区别在于前者用于进程间通讯，后者用于本地设备分配。
'''
# 是指在一台机器上(一个node上)进程的相对序号，例如机器一上有0,1,2,3,4,5,6,7，机器二上也有0,1,2,3,4,5,6,7。local_rank在node之间相互独立
# 进程内，GPU 编号，非显式参数，由 torch.distributed.launch 内部指定。比方说， rank = 3，local_rank = 0 表示第 3 个进程内的第 1 块 GPU
# -本地序号。这个 Worker 是这台机器上的第几个 Worker
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
# rank是指在整个分布式任务中进程的序号；表示进程序号，用于进程间通讯，表征进程优先级。rank = 0 的主机为 master 节点。
# -进程序号。这个 Worker 是全局第几个 Worker
RANK = int(os.getenv("RANK", -1))
# 全局进程总个数，即在一个分布式任务中rank的数量
# 总共有几个 Worker
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
# 检查git信息
GIT_INFO = check_git_info()

''' =====================1.载入参数和初始化配置信息==========================  '''
'''
1.1 载入参数
  hyp,  # 超参数 可以是超参数配置文件的路径或超参数字典 path/to/hyp.yaml or hyp
  opt,  # main中opt参数
  device,  # 当前设备
  callbacks  # 用于存储Loggers日志记录器中的函数，方便在每个训练阶段控制日志的记录情况
'''


def train(hyp, opt, device, callbacks):
    """
    基本信息配置
    Trains a YOLOv5 model on a custom dataset using specified hyperparameters, options, and device, managing datasets,
    model architecture, loss computation, and optimizer steps.

    Args:
        hyp (str | dict): Path to the hyperparameters YAML file or a dictionary of hyperparameters.
        opt (argparse.Namespace): Parsed command-line arguments containing training options.
        device (torch.device): Device on which training occurs, e.g., 'cuda' or 'cpu'.
        callbacks (Callbacks): Callback functions for various training events.

    Returns:
        None

    Models and datasets download automatically from the latest YOLOv5 release.

    Example:
        Single-GPU training:
        ```bash
        $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
        $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
        ```

        Multi-GPU DDP training:
        ```bash
        $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights
        yolov5s.pt --img 640 --device 0,1,2,3
        ```

        For more usage details, refer to:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    # 解析opt传入的参数
    # 从opt获取参数。日志保存路径，轮次、批次、权重、进程序号(主要用于分布式训练)等
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    callbacks.run("on_pretrain_routine_start")

    '''
        1.2 创建训练权重目录，设置模型、txt等保存的路径
    '''
    # Directories 获取记录训练日志的保存路径

    # 设置保存权重路径 如runs/train/exp1/weights
    w = save_dir / "weights"  # weights dir
    # 创建保存训练结果的文件夹
    # 新建文件夹 weights train evolve
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    # 保存训练结果的目录，如：，runs/train/exp1/weights/last.pt
    # 保存训练结果的目录，如last.pt和best.pt
    last, best = w / "last.pt", w / "best.pt"

    '''
        1.3 读取hyp(超参数)配置文件
    '''
    # Hyperparameters 加载超参数
    # isinstance()是否是已知类型。 判断hyp是字典还是字符串
    if isinstance(hyp, str):
        # 若hyp是字符串，即认定为路径，则加载超参数为字典
        with open(hyp, errors="ignore") as f:
            # 加载yaml文件
            hyp = yaml.safe_load(f)  # load hyps dict 加载超参信息
    # 打印超参数，彩色字体
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    '''
        1.4 设置参数的保存路径
    '''
    # Save run settings 保存训练中的参数hyp和opt
    if not evolve:
        # 保存超参数为yaml配置文件
        yaml_save(save_dir / "hyp.yaml", hyp)
        # 保存命令行参数为yaml配置文件
        yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers
    # 定义数据集字典
    data_dict = None

    '''
        1.5 加载相关日志功能:如tensorboard,logger,wandb
    '''
    # Loggers 设置wandb和tb两种日志, wandb和tensorboard都是模型信息，指标可视化工具
    if RANK in {-1, 0}:
        # 如果进程编号为-1或0

        include_loggers = list(LOGGERS)
        if getattr(opt, "ndjson_console", False):
            include_loggers.append("ndjson_console")
        if getattr(opt, "ndjson_file", False):
            include_loggers.append("ndjson_file")

        # 初始化日志记录器实例
        loggers = Loggers(
            save_dir=save_dir,
            weights=weights,
            opt=opt,
            hyp=hyp,
            logger=LOGGER,
            include=tuple(include_loggers),
        )

        # Register actions
        for k in methods(loggers):
            # 将日志记录器中的方法与字符串进行绑定
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    '''
        1.6 配置:画图开关,cuda,种子,读取数据集相关的yaml文件
    '''
    # Config 画图
    # 是否绘制训练、测试图片、指标图等，使用进化算法则不绘制
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != "cpu"
    # 随机种子
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # 加载数据配置信息
    # torch_distributed_zero_first 同步所有进程
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(
            data)  # check if None check_dataset 检查数据集，如果没找到数据集则下载数据集(仅适用于项目中自带的yaml文件数据集)
    # 获取训练集和验证机的路径
    train_path, val_path = data_dict["train"], data_dict["val"]
    # 设置类别，判断是否为单类
    # nc：数据集有多少种类别
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    # 类别对应的名称
    # names: 数据集所有类别的名字，如果设置了single_cls则为一类
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    # 判断是否是coco数据集
    # 当前数据集是否是coco数据集(80个类别)
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset

    ''' =====================2.model：加载网络模型==========================  '''
    # Model 载入模型
    # 检查文件后缀是否是.pt
    check_suffix(weights, ".pt")  # check weights
    # 加载预训练权重 yolov5提供了5个不同的预训练权重，可以根据自己的模型选择预训练权重
    pretrained = weights.endswith(".pt")
    '''
        2.1预训练模型加载 
    '''
    if pretrained:
        # 使用预训练的话：
        # torch_distributed_zero_first(RANK): 用于同步不同进程对数据读取的上下文管理器
        with torch_distributed_zero_first(LOCAL_RANK):
            # 如果不存在就从网站上下载
            # 如果本地不存在就从google云盘中自动下载模型
            # 通常会下载失败，建议提前下载下来放进weights目录
            weights = attempt_download(weights)  # download if not found locally
        # ============加载模型以及参数================= #
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        """
            两种加载模型的方式：opt.cfg / ckpt['model'].yaml
            这两种方式的区别：区别在于是否是使用resume
            使用resume-断点训练：将opt.cfg设为空，选择ckpt['model'].yaml 创建模型，且不加载anchor
            这也影响了下面是否除去anchor的key(也就是不加载anchor), 如果resume则不加载anchor
            原因：
                使用断点训练时,保存的模型会保存anchor,所以不需要加载，
                主要是预训练权重里面保存了默认coco数据集对应的anchor，
                如果用户自定义了anchor，再加载预训练权重进行训练，会覆盖掉用户自定义的anchor。
        """
        # ***加载模型***
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        # 以下三行是获得anchor
        # 若cfg 或 hyp.get('anchors')不为空且不使用中断训练 exclude=['anchor'] 否则 exclude=[]
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        # 筛选字典中断点键值对，把exclude删除
        # 将预训练模型中的所有参数保存下来，赋值给csd
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        # 判断预训练参数和新创建的模型参数有多少是相同的
        # 筛选字典中的键值对，把exclude删除
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # 模型创建
        model.load_state_dict(csd, strict=False)  # load
        # 显示加载预训练权重的的键值对和创建模型的键值对
        # 如果pretrained为ture 则会少加载两个键对（anchors, anchor_grid）
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        # 不使用预训练权重
        # #直接加载模型，ch为输入图片通道
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    amp = check_amp(model)  # check AMP

    '''
        2.2 冻结层
    '''
    # Freeze 冻结训练的网络层
    """
    冻结模型层,设置冻结层名字即可
    作用：冰冻一些层，就使得这些层在反向传播的时候不再更新权重,需要冻结的层,可以写在freeze列表中
    freeze为命令行参数，默认为0，表示不冻结
    """
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # 首先遍历所有层
    for k, v in model.named_parameters():
        # 为所有层的参数设置梯度
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        # 判断是否需要冻结
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            # 冻结的训练层梯度不更新
            v.requires_grad = False

    # Image size 设置训练和测试图片尺寸
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # 检查输入图片分辨率是否能被32整除
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size 设置一次训练所选取的样本数
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        # 确保batch size满足要求
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    '''
        2.3 优化器设置
    '''
    # Optimizer 优化器设置
    nbs = 64  # nominal batch size
    """
     # Optimizer 优化器设置
     nbs = 64
        batchsize = 16
        accumulate = 64 / 16 = 4
        模型梯度累计accumulate次之后就更新一次模型 相当于使用更大batch_size
    """
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 权重衰减参数
    # 根据accumulate设置权重衰减参数，防止过拟合
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    # 打印缩放后的权重衰减超参数
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    '''
        2.4 学习率设置
    '''
    # Scheduler 设置学习率策略:两者可供选择，线性学习率和余弦退火学习率
    if opt.cos_lr:
        # 是否使用余弦学习率调整方式
        # 使用余弦退火学习率
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:
        # 使用线性学习率
        def lf(x):
            """Linear learning rate scheduler function with decay calculated by epoch proportion."""
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear

    # 可视化 scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    '''
        2.5 训练前最后准备
    '''
    # 若进程编号为-1或0
    # EMA 对模型的参数做平均，给与近期数据更高权重的平均方法
    # EMA 设置ema（指数移动平均），考虑历史值对参数的影响，目的是为了收敛的曲线更加平滑
    # 为模型创建EMA指数滑动平均,如果GPU进程数大于1,则不创建
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume 断点续训
    # 断点续训其实就是把上次训练结束的模型作为预训练模型，并从中加载参数
    best_fitness, start_epoch = 0.0, 0
    # 如果有预训练
    if pretrained:
        if resume:
            # 获取数据
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        # 将预训练的相关参数从内存中删除
        del ckpt, csd

    # DP mode 使用单机多卡模式训练，目前一般不使用
    # 单机多卡
    # rank为进程编号。如果rank=-1且gpu数量>1则使用DataParallel单机多卡模式，效果并不好（分布不平均）
    # rank=-1且gpu数量=1时,不会进行分布式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        # 将数据分割成多个部分，然后在不同的GPU上并行处理这些数据部分。每个GPU都运行一个模型的副本，并处理一部分输入数据。
        # 最后，所有GPU上的结果将被收集并合并，以产生与单个GPU上运行模型相同的输出
        # 主要用于多卡的GPU服务器，使用这个函数来用多个GPU来加速训练
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    # 多卡归一化
    if opt.sync_bn and cuda and RANK != -1:
        # 多卡训练，把不同卡的数据做个同步
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    ''' =====================3.加载训练数据集==========================  '''
    '''
        3.1 创建数据集
    '''
    # Trainloader 训练集数据加载
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=opt.seed,
    )
    '''
      返回一个训练数据加载器，一个数据集对象:
      训练数据加载器是一个可迭代的对象，可以通过for循环加载1个batch_size的数据
      数据集对象包括数据集的一些参数，包括所有标签值、所有的训练数据路径、每张图片的尺寸等等
    '''
    # 统计dataset的label信息
    labels = np.concatenate(dataset.labels, 0)
    # 标签编号最大值
    # mlc标签编码最大值
    mlc = int(labels[:, 0].max())  # max label class
    # 如果小于类别数则表示有问题
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0 验证集数据集加载
    # 验证集数据加载
    if RANK in {-1, 0}:
        # 若进程编号为-1或0

        # 加载验证集数据加载器
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            prefix=colorstr("val: "),
        )[0]

        # 不使用断点训练
        if not resume:
            '''
                3.2 计算anchor
            '''
            # Anchors 计算默认锚框anchor与数据集标签框的高宽比
            if not opt.noautoanchor:
                # dataset是在上边创建train_loader时生成的
                # hyp['anchor_t']是从配置文件hpy.scratch.yaml读取的超参数 anchor_t:4.0
                # 当配置文件中的anchor计算bpr（best possible recall）小于0.98时才会重新计算anchor
                # best possible recall最大值1，如果bpr小于0.98，程序会根据数据集的label自动学习anchor的尺寸
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
                '''
                    参数dataset代表的是训练集，hyp['anchor_t']是从配置文件hpy.scratch.yaml读取的超参数，anchor_t:4.0
                    当配置文件中的anchor计算bpr（best possible recall）小于0.98时才会重新计算anchor。
                    best possible recall最大值1，如果bpr小于0.98，程序会根据数据集的label自动学习anchor的尺寸
                '''
            # 半精度
            model.half().float()  # pre-reduce anchor precision

        # 在每个训练前例行程序结束时触发所有已注册的回调
        callbacks.run("on_pretrain_routine_end", labels, names)

    # DDP mode 如果rank不等于-1,则使用DistributedDataParallel模式
    if cuda and RANK != -1:
        # local_rank为gpu编号,rank为进程,例如rank=3，local_rank=0 表示第 3 个进程内的第 1 块 GPU
        model = smart_DDP(model)

    ############4.训练#######################################################
    '''
        4.1 初始化训练需要的模型参数
    '''
    # Model attributes 根据自己数据集的类别数和网络FPN层数设置各个损失的系数
    # smart_DDP和de_parallel代码在utils.torch_utils中
    # 对hpy字典中的一些值进行缩放和预设置,以适应不同的层级、类别、图像尺寸和标签平滑需求
    # 默认 nl = 3
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    # hyp-low中给出的 box=0.05; cls=0.5; obj=1.0
    # hyp['box'] = 0.05*3/3=0.05
    # box为预测框的损失
    hyp["box"] *= 3 / nl  # scale to layers
    # hyp['cls'] = 0.5*20/80*3/3=0.125
    # cls为分类的损失
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    # hyp['obj']=1.0*(640/640)**2*3/nl=1
    # obj为置信度损失
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    # 标签平滑
    hyp["label_smoothing"] = opt.label_smoothing
    # 设置模型的类别，然后将检测的类别个数保存到模型
    model.nc = nc  # attach number of classes to model
    # 设置模型的超参数，然后将超参数保存到模型
    model.hyp = hyp  # attach hyperparameters to model
    # 从训练样本标签得到类别权重（和类别中的目标数即类别频率成反比）
    # 从训练的样本标签得到类别权重，然后将类别权重保存至模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # 获取类别的名字，然后将分类标签保存至模型
    model.names = names

    '''
        4.2 训练热身部分
    '''
    # Start training
    # 获取当前时间
    t0 = time.time()
    # 类别总数
    nb = len(train_loader)  # number of batches
    # 获取热身训练的迭代次数
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    # 初始化maps(每个类别的map)和results
    # 初始化 map和result
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    # 设置学习率衰减所进行到的轮次,即使打断训练,使用resume接着训练也能正常衔接之前的训练进行学习率衰减
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 设置amp混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # 早停止，不更新结束训练
    stopper, stop = EarlyStopping(patience=opt.patience), False
    # 初始化损失函数
    compute_loss = ComputeLoss(model)  # init loss class
    # 执行训练方法
    callbacks.run("on_train_start")
    # 打印日志输出信息
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} val\n'  # 打印训练和测试输入图片分辨率
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'  # 加载图片时调用的cpu进程数
        f"Logging results to {colorstr('bold', save_dir)}\n"  # 日志目录
        f'Starting training for {epochs} epochs...'  # 从哪个epoch开始训练
    )

    '''
        4.3 开始训练
    '''
    # 正式开始训练
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        '''
            告诉模型现在是训练阶段 因为BN层、DropOut层、两阶段目标检测模型等
            训练阶段阶段和预测阶段进行的运算是不同的，所以要将二者分开
            model.eval()指的是预测推断阶段
        '''
        model.train()

        # Update image weights (optional, single-GPU only) 更新图片的权重
        """
        如果设置图片采样策略
        则根据前面初始化的图片采样权重model.class_weights以及maps配合每张图片包含的类别数
        通过random.choices生成图片所有indices从而进行采样
        """
        if opt.image_weights:  # 获取图片采样的权重
            # 经过一轮训练，若哪一类的不精确度高，那么这个类就会被分配一个较高的权重，来增加它被采样的概率
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            # 将计算出的权重换算到图片的维度，将类别的权重换算为图片的权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            # 通过random.choices生成图片索引indices从而进行采样，这时图像会包含一些难识别的样本
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(3, device=device)  # mean losses
        # 分布式训练的设置
        # DDP模式打乱数据，并且dpp.sampler的随机采样数据是基于epoch+seed作为随机种子，每次epoch不同，随机种子不同
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        # 将训练数据迭代器做枚举，可以遍历出索引值
        pbar = enumerate(train_loader)
        # 训练参数的表头
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            # 若进程编号为-1或0

            # 进度条显示
            # 通过tqdm创建进度条，方便训练信息的展示
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        # 将优化器中的所有参数梯度设为0
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")
            # ni: 计算当前迭代次数 iteration
            ni = i + nb * epoch  # number integrated batches (since train start)
            # 将图片加载至设备 并做归一化
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup 热身训练
            """
            热身训练(前nw次迭代 &  # xff0c;一般是3)
            在前nw次迭代中 &  # xff0c;根据以下方式选取accumulate和学习率
            """
            '''
                热身训练(前nw次迭代),热身训练迭代的次数iteration范围[1:nw] 
                在前nw次迭代中, 根据以下方式选取accumulate和学习率
            '''
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                # 遍历优化器中的所有参数组
                for j, x in enumerate(optimizer.param_groups):
                    """
                    bias的学习率从0.1下降到基准学习率lr*lf(epoch),
                        其他的参数学习率从0增加到lr*lf(epoch).
                        lf为上面设置的余弦退火的衰减函数
                        动量momentum也从0.9慢慢变到hyp['momentum'](default=0.937)
                    """
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale 设置多尺度训练，从imgsz * 0.5, imgsz * 1.5 + gs随机选取尺寸
            # imgsz: 默认训练尺寸   gs: 模型最大stride=32   [32 16 8]
            if opt.multi_scale:  # 随机改变图片的尺寸
                # Multi-scale
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    # 下采样
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward 前向传播
            with torch.cuda.amp.autocast(amp):
                # 将图片送入网络得到一个预测结果
                pred = model(imgs)  # forward
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    # 平均不同gpu之间的梯度
                    # 采用DDP训练,平均不同gpu之间的梯度
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    # 如果采用collate_fn4取出mosaic4数据loss也要翻4倍
                    loss *= 4.0

            # Backward 反向传播 scale为使用自动混合精度运算
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            # 模型会对多批数据进行累积，只有达到累计次数的时候才会更新参数，再还没有达到累积次数时 loss会不断的叠加 不会被新的反传替代
            # 模型反向传播accumulate次之后再根据累计的梯度更新一次参数
            if ni - last_opt_step >= accumulate:
                '''
                 scaler.step()首先把梯度的值unscale回来，
                 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                 否则，忽略step调用，从而保证权重不更新（不被破坏）
                '''
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step 参数更新
                # 更新参数
                scaler.update()
                # 完成一次累积后，再将梯度清零，方便下一次清零
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                # 计数
                last_opt_step = ni

            # Log 打印Print一些信息 包括当前epoch、显存、损失(box、obj、cls、total)、当前batch的target的数量和图片的size等信息
            if RANK in {-1, 0}:
                # 若进程编号为-1或0

                # 打印显存，进行的轮次，损失，target的数量和图片的size等信息
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                # 计算显存
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                # 进度条显示以上信息
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                # 调用Loggers中的on_train_batch_end方法，将日志记录并生成一些记录的图片
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler 进行学习率衰减
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        # 根据前面设置的学习率更新策略更新学习率
        scheduler.step()

        '''
            4.4 训练完成保存模型  
        '''
        if RANK in {-1, 0}:
            # 若进程编号为-1或0

            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            # 把model中的属性赋值给ema
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            # 判断当前epoch是否是最后一轮
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop

            # notest: 是否只测试最后一轮  True: 只测试最后一轮   False: 每轮训练完都测试mAP
            if not noval or final_epoch:  # Calculate mAP
                # 测试使用的是ema(对模型的参数做平均)模型
                # verbose设置为true后，每轮的验证都输出每个类别的信息
                """
                    测试使用的是ema（指数移动平均 对模型的参数做平均）的模型
                       results: [1] Precision 所有类别的平均precision(最大f1时)
                                [1] Recall 所有类别的平均recall
                                [1] map@0.5 所有类别的平均mAP@0.5
                                [1] map@0.5:0.95 所有类别的平均mAP@0.5:0.95
                                [1] box_loss 验证集回归损失, obj_loss 验证集置信度损失, cls_loss 验证集分类损失
                       maps: [80] 所有类别的mAP@0.5:0.95
                """
                results, maps, _ = validate.run(
                    data_dict,  # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
                    batch_size=batch_size // WORLD_SIZE * 2,  # 要保证batch_size能整除卡数
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,  # 是否是单类数据集
                    dataloader=val_loader,
                    save_dir=save_dir,  # 保存地址 runs/train/expn
                    plots=False,  # 是否可视化
                    callbacks=callbacks,
                    compute_loss=compute_loss,  # 损失函数(train)
                )

            # Update best mAP 更新best_fitness
            #  fi: [P, R, mAP@.5, mAP@.5-.95]的一个加权值 = 0.1*mAP@.5 + 0.9*mAP@.5-.95
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            # 若当前的fitness大于最佳的fitness
            if fi > best_fitness:
                # 将最佳fitness更新为当前fitness
                best_fitness = fi
            # 保存验证结果
            log_vals = list(mloss) + list(results) + lr
            # 记录验证数据
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

            # Save model 保存模型
            """
            保存带checkpoint的模型用于inference或resuming training
                保存模型的同时还保存epoch，results，optimizer等信息
                optimizer在最后一轮不会报错
                model保存的是EMA后的模型
            """
            if (not nosave) or (final_epoch and not evolve):  # if save
                # 将当前训练过程中的所有参数赋值给ckpt
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete 保存每轮的模型
                torch.save(ckpt, last)
                # 如果这个模型的fitness是最佳的
                if best_fitness == fi:
                    # 保存这个最佳的模型
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                # 模型保存完毕 将变量从内存中删除
                del ckpt
                # 记录保存模型时的日志
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping：提前停止，不更新结束训练
        # DDP 是一个支持多机多卡、分布式训练的深度学习工程方法
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    '''
        4.5 打印信息并释放显存 
    '''
    # 打印一些信息
    if RANK in {-1, 0}:
        # 若进程编号为-1或0

        # 训练停止 向控制台输出信息
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        # 可视化训练结果: results1.png   confusion_matrix.png 以及('F1', 'PR', 'P', 'R')曲线变化  日志信息
        for f in last, best:
            if f.exists():
                # 模型训练完后, strip_optimizer函数将optimizer从ckpt中删除
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    # 把最好的模型在验证集上跑一边 并绘图
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots
                    if is_coco:
                        # 如果是coco数据集
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        # 记录训练终止时的日志
        callbacks.run("on_train_end", last, best, epoch, results)

    # 释放显存
    torch.cuda.empty_cache()
    return results


"""
常用的配置：
//--weights: 模型文件
//--data: 数据集配置文件 包括path、train、val、test、nc、names、download等
//--epochs: 总训练轮次
//--batch-size: 每次传递多少张图给GPU，所有GPU的总批处理大小，自动批处理为-1
//--imgsz: 输入网络的图片分辨率大小，+-30%
//--resume: 断点续训，从上次打断的训练结果处接着训练 默认False
//--device: 设备选择，如果是GPU就输入GPU索引[如，1，2..]，CPU训练就填cpu
//--workers: 数据加载过程中使用的线程数量，根据自己的电脑设置
//--patience: 早停策略是一种常用的防止过拟合的方法
"""


# =============================================三、设置opt参数==================================================='''
def parse_opt(known=False):
    """
    Parses command-line arguments for YOLOv5 training, validation, and testing.

    Args:
        known (bool, optional): If True, parses known arguments, ignoring the unknown. Defaults to False.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Example:
        ```python
        from ultralytics.yolo import parse_opt
        opt = parse_opt()
        print(opt)
        ```

    Links:
        Models: https://github.com/ultralytics/yolov5/tree/master/models
        Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    parser = argparse.ArgumentParser()
    # 模型文件
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    # 模型配置
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    # 数据集配置文件 包括path、train、val、test、nc、names、download等
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    # 初始超参文件初始超参文件
    # hpy超参数设置文件（lr/sgd/mixup）./data/hyps/下面有5个超参数设置文件，每个文件的超参数初始值有细微区别，用户可以根据自己的需求选择其中一个
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    # 总训练轮次， 默认轮次为300次
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    # 每次传递多少张图给GPU，所有GPU的总批处理大小，自动批处理为-1（这里配置的是单个GPU的数量，多个GPU时，乘以gpu数量）
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    # 输入网络的图片分辨率大小(
    # 1.这里没有分别传入宽高值，只有一个值，表示宽高一致，即为正方形；
    # 2.填写参数时，要根据当前图片分辨率(如：800*600)，32的倍数，上下不要相差超过百分之三十至四十。
    #   图片很大，但设置的值很小时，图片传入网络时，会被压缩到当前配置的正方形尺寸，会造成丢失像素，导致训练效果不好
    #   比如：图片宽1280，配置的是640，图片就会被压缩到640，相当于缩小了百分之五十((1280-640)/1280),不合适，可以适当调高。
    #   比如：图片宽800，配置的是640，图片就会被压缩到640，相当于缩小了百分之二十((800-640)/800))，合适。
    #   备注：计算时，使用图像的长边值为基准计算
    # )
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    # 是否进行矩形训练，即不失真的resize图像训练，默认为False；用的少
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    # 断点续训，从上次打断的训练结果处接着训练 黑认False
    # resume: 是否接着上次的训练结果，继续训练
    # 矩形训练：将比例相近的图片放在一个batch（由于batch里面的图片shape是一样的）
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    # 设置为True后只保存最后一个epoch权重；默认每次都保存，默认策略比较稳当，一旦中途中断，将会丢失权重，需要重新训练
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    # 设置为True后只测试最后一个epoch
    # noval: 最后进行测试, 设置了之后就是训练结束都测试一下， 不设置每轮都计算mAP, 建议不设置
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    # 禁用anchors 默认False(自动调整anchors)models.yaml
    # noautoanchor: 不自动调整anchor, 默认False, 自动调整anchor
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    # 不保存打印文件
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    # 是否进行超参进化（寻找最优参数的方式） 默认False
    # evolve: 参数进化， 遗传算法调参
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    #
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    # 断点训练+超参数优化
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    # 谷歌云盘bucket 一般用不到
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    # 设置为True，提前缓存图像可用于加速训练，默认为False
    # cache: 是否提前缓存图片到内存，以加快训练速度，默认False
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    # 加权图像选择进行训练，默认为False;使用图片采样策略，默认不使用
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    # 设备选择，如果是GPU就输入GPU索引[如，1，2..]，CPU训练就填cpu
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 是否采用多尺度训练，默认为False；img--size参数下，加减50%变化，进行训练。如果有同一图像缩放变化的需求时，可以尝试调整；使用这个配置时，训练效率会降低，根据实际需求调整
    # 是否进行多尺度训练
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    # 数据集是单类别还是多类别，默认False
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    # 优化器，负责更新网络参数，使得网络能更好地拟合数据集，可选:SGD，"Adam”，"AdamW'
    # 用处较大，不同优化器训练效果不一样，
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    # 是否使用跨卡(多GPU卡片同步训练)同步BN，在DDP模式使用，默认Fase，当使用sync bn时，需要将该参数设置为True，使用的少
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    # 数据加载过程中使用的线程数量，根据自己的电脑设置
    # dataloader的最大worker数量 （使用多线程加载图片）
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    # 训练模型保存的位置，默认为runs/train，保持默认即可
    # 训练结果的保存路径
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    # 保存项目名字，一般是run/train/exp，默认即可
    # 训练结果的文件名称
    parser.add_argument("--name", default="exp", help="save to project/name")
    # 模型目录是否存在，不存在就创建，默认为False
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 默认值:False，表示使用的矩形锚框，使用四边可以提高模型的检测精度，但同时也会增加计算量和内存占用，根据自己的电脑设器
    # 四元数据加载器: 允许在较低 --img 尺寸下进行更高 --img 尺寸训练的一些好处。
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    # 设置学习率调整策略，即使用余弦退火策略调整学习率，默认: false
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    # 设置标签平滑,取值: 0~1
    # 标签平滑 / 默认不增强， 用户可以根据自己标签的实际情况设置这个参数，建议设置小一点 0.1 / 0.05
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    # 早停策略是一种常用的防止过拟合的方法，通过监控模型在验证集上的性能，当模型的性能在连续若于个epoch中没有明显提升时，
    # 就停止训练，一般来说，patience的认值为1，即每个epoch都会检查模型在验证集上的性能，如果连续一个epoch没有明显提升，就停止训练。如果patience的值为更大，则表示
    # 更大的耐心值，需要更长的训练时间。
    # 需要注意的是，patience参数的使用需要在训练前设定一个合适的验证集，以便监控模型在训练过程中的性能变化。同时，如果数据集较小或者模型复杂度较高
    # patience参数可能需要设置为较小的值以防止过拟合。
    # 早停止耐心次数 / 100次不更新就停止训练
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    # 设置冻结层
    # --freeze冻结训练 可以设置 default = [0] 数据量大的情况下，建议不设置这个参数
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    # 设置模型保存的周期，每隔多少个epoch保存一次模型，默认每一次，保存一次模型
    # --save-period 多少个epoch保存一下checkpoint
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    # 全局随机种子，随机种子也可以与其他参数一起使用，如数据集的随机裁剪和翻转等，以获得更好的训练效果
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    # 设置分布式训练中的本地排名，多机训练时使用
    # --local_rank 进程编号 / 多卡使用
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments 记录器参数
    # 设置wandB库中的实体
    # 在线可视化工具，类似于tensorboard工具
    parser.add_argument("--entity", default=None, help="Entity")
    # 用于上传数据集作为WandB的artifact table
    # upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    # 设置每个保存周期中bboxlog的间隔
    # bbox_interval: 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    # 设置wandB用于为上传的模型设置别名
    # 使用数据的版本
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    # 作用就是当仅获取到基本设置时，如果运行命令中传入了之后才会获取到的其他配置，不会报错；而是将多出来的部分保存起来，留到后面使用
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """
    Runs training or hyperparameter evolution with specified options and optional callbacks.

    Args:
        opt (argparse.Namespace): The command-line arguments parsed for YOLOv5 training and evolution.
        callbacks (ultralytics.utils.callbacks.Callbacks, optional): Callback functions for various training stages.
            Defaults to Callbacks().

    Returns:
        None

    Note:
        For detailed usage, visit:
        https://github.com/ultralytics/yolov5/tree/master/models
    """
    '''
       2.1  检查分布式训练环境
    '''
    # 若进程编号为-1或0
    if RANK in {-1, 0}:
        # 输出所有训练参数 / 参数以彩色的方式表现
        print_args(vars(opt))
        # 检查代码版本是否更新
        # 检测YOLO v5的github仓库是否更新，若已更新，给出提示
        check_git_status()
        # 检查所需要的包是否都安装了
        # 检查requirements.txt所需包是否都满足
        check_requirements(ROOT / "requirements.txt")

    # Resume (from specified or most recent last.pt)
    '''
        2.2  判断是否断点续训
    '''
    # 断点训练
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        # isinstance()是否是已经知道的类型
        # 如果resume是True，则通过get_latest_run()函数找到runs为文件夹中最近的权重文件last.pt
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        # opt.yaml是训练时的命令行参数文件
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        # 把opt的参数替换为last.pt中opt的参数
        # 判断是否为文件，若不是文件抛出异常
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        # 超参数替换，将训练时的命令行参数加载进opt参数对象中
        opt = argparse.Namespace(**d)  # replace
        # opt.cfg设置为'' 对应着train函数里面的操作(加载权重时是否加载权重里的anchor)
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        # 不使用断点训练，就从文件中读取相关参数
        # 加载参数
        # check_file （utils/general.py）的作用为查找/下载文件 并返回该文件的路径。
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        # 如果模型文件和权重文件为空，弹出警告
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        # 如果要进行超参数进化，重建保存路径
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                # 设置新的项目输出目录
                opt.project = str(ROOT / "runs/evolve")
            # 将resume传递给exist_ok
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        # 保存相关信息到文件中
        # 根据opt.project生成目录，并赋值给opt.save_dir  如: runs/train/exp1
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    '''
        2.3  判断是否分布式训练
    '''
    # DDP mode 是一个支持多机多卡、分布式训练的深度学习工程方法
    # 选择device
    device = select_device(opt.device, batch_size=opt.batch_size)
    # 当进程内的GPU编号不为-1时，才会进入DDP
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        # 不能使用图片采样策略
        assert not opt.image_weights, f"--image-weights {msg}"
        # 不能使用超参数进化
        assert not opt.evolve, f"--evolve {msg}"
        # 分布式训练时批次数，不能为-1(必须指定)
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        # WORLD_SIZE表示全局的进程数
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        #  用于DDP训练的GPU数量不足
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"

        # 可指定多卡
        # 设置装载程序设备
        torch.cuda.set_device(LOCAL_RANK)
        # 保存装载程序的设备
        device = torch.device("cuda", LOCAL_RANK)
        # 初始化多进程
        # torch.distributed是用于多GPU训练的模块
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)
        )

    '''
        2.4  判断是否进化训练
    '''
    # Train 训练模式: 如果不进行超参数进化，则直接调用train()函数，开始训练
    # 如果不使用超参数进化
    if not opt.evolve:
        # 开始训练
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # 遗传进化算法，边进化边训练

        # Hyperparameter evolution metadata (including this hyperparameter True-False, lower_limit, upper_limit) 超参数演化元数据
        # 超参数列表(包括此超参数是否参与进化，下限，下限)
        meta = {
            "lr0": (False, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3) 初始学习率
            "lrf": (False, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf) 循环学习率
            "momentum": (False, 0.6, 0.98),  # SGD momentum/Adam beta1 学习率动量
            "weight_decay": (False, 0.0, 0.001),  # optimizer weight decay 权重衰减系数
            "warmup_epochs": (False, 0.0, 5.0),  # warmup epochs (fractions ok) 预热学习
            "warmup_momentum": (False, 0.0, 0.95),  # warmup initial momentum 预热学习动量
            "warmup_bias_lr": (False, 0.0, 0.2),  # warmup initial bias lr 预热初始学习率
            "box": (False, 0.02, 0.2),  # box loss gain iou损失系数
            "cls": (False, 0.2, 4.0),  # cls loss gain  cls损失系数
            "cls_pw": (False, 0.5, 2.0),  # cls BCELoss positive_weight 正样本权重
            "obj": (False, 0.2, 4.0),  # obj loss gain (scale with pixels) 有无物体系数
            "obj_pw": (False, 0.5, 2.0),  # obj BCELoss positive_weight 有无物体BCELoss正样本权重
            "iou_t": (False, 0.1, 0.7),  # IoU training threshold IoU训练时的阈值
            "anchor_t": (False, 2.0, 8.0),  # anchor-multiple threshold anchor的长宽比（长:宽 = 4:1）
            "anchors": (False, 2.0, 10.0),  # anchors per output grid (0 to ignore) 每个输出层的anchors数量(0 to ignore)
            # 以下系数是数据增强系数&#xff0c;包括颜色空间和图片空间
            "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # image HSV-Hue augmentation (fraction) 色调
            "hsv_s": (True, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction) 饱和度
            "hsv_v": (True, 0.0, 0.9),  # image HSV-Value augmentation (fraction) 亮度
            "degrees": (True, 0.0, 45.0),  # image rotation (+/- deg) 旋转角度
            "translate": (True, 0.0, 0.9),  # image translation (+/- fraction) 平移
            "scale": (True, 0.0, 0.9),  # image scale (+/- gain) 图像缩放
            "shear": (True, 0.0, 10.0),  # image shear (+/- deg) 图像裁剪
            "perspective": (True, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001 透明度
            "flipud": (True, 0.0, 1.0),  # image flip up-down (probability) 进行上下翻转概率
            "fliplr": (True, 0.0, 1.0),  # image flip left-right (probability) 进行左右翻转概率
            "mosaic": (True, 0.0, 1.0),  # image mixup (probability) 进行Mosaic概率
            "mixup": (True, 0.0, 1.0),  # image mixup (probability) 进行图像混叠概率(即，多张图像重叠在一起)
            "copy_paste": (True, 0.0, 1.0),  # 复制粘贴增强的概率
        }  # segment copy-paste (probability)

        # GA configs
        # 遗传算法的配置

        # 种群大小
        pop_size = 50
        # 变异率的最小值和最大值
        mutation_rate_min = 0.01
        mutation_rate_max = 0.5
        # 交叉率的最小值和最大值
        crossover_rate_min = 0.5
        crossover_rate_max = 1
        # 精英大小(保留的最好个体数量)的最小值和最大值
        min_elite_size = 2
        max_elite_size = 5
        # 锦标赛大小(用于选择附带的选择池大小)的最小值和最大值
        tournament_size_min = 2
        tournament_size_max = 10

        # 从指定文件超参文件，加载默认超参数
        with open(opt.hyp, errors="ignore") as f:
            # 通过yaml工具构建为hyp对象
            hyp = yaml.safe_load(f)  # load hyps dict
            # 如果超参数文件中没有'anchors'这个超参数，则设为3
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        # 不使用AutoAnchors
        if opt.noautoanchor:
            # 从GA种群中删除
            del hyp["anchors"], meta["anchors"]
        # 使用进化算法时，仅在最后的epoch测试和保存
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        # 拼接保存路径
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            """
                遗传算法调参：遵循适者生存、优胜劣汰的法则，即寻优过程中保留有用的，去除无用的。
                遗传算法需要提前设置4个参数: 群体大小/进化代数/交叉概率/变异概率
            """
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )

        # Delete the items in meta dictionary whose first value is False
        # 删除元字典中第一个值为False的项-->不参与进化的参数都删除掉
        del_ = [item for item, value_ in meta.items() if value_[0] is False]
        # 在删除之前备份一下
        hyp_GA = hyp.copy()  # Make a copy of hyp dictionary
        # 开始删除不参与进化的超参数
        for item in del_:
            del meta[item]  # Remove the item from meta dictionary
            del hyp_GA[item]  # Remove the item from hyp_GA dictionary

        # Set lower_limit and upper_limit arrays to hold the search space boundaries
        # 设置lower_limit和upper_limit数组以保持搜索空间边界
        # 获取所有超参数项的下限列表
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
        # 获取所有超参数项的上限列表
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])

        # Create gene_ranges list to hold the range of values for each gene in the population
        # 创建gene_ranges列表，以保存群体中每个超参属性的值范围
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]

        # Initialize the population with initial_values or random values
        # 初始化种群，使用初始化值或随机值
        initial_values = []

        # If resuming evolution from a previous checkpoint
        # 根据之前的ckpt继续进化
        # 断点训练+超参数优化
        if opt.resume_evolve is not None:
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
            # 打开文件
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:
                evolve_population = yaml.safe_load(f)
                for value in evolve_population.values():
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # If not resuming from a previous checkpoint, generate initial values from .yaml files in opt.evolve_population
        # 如果不是从之前的ckpt恢复，则从opt.evolve_population中的yaml文件生成初始值
        else:
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]
            for file_name in yaml_files:
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
                    value = yaml.safe_load(yaml_file)
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # Generate random values within the search space for the rest of the population
        # 为种群中剩余的部分在搜索空间内生成随机值
        if initial_values is None:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
        elif pop_size > 1:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
            for initial_value in initial_values:
                population = [initial_value] + population

        # Run the genetic algorithm for a fixed number of generations
        # 对固定的一代数运行遗传算法
        list_keys = list(hyp_GA.keys())

        # 选择超参数的遗传迭代次数 默认为迭代300次
        for generation in range(opt.evolve):
            if generation >= 1:
                save_dict = {}
                for i in range(len(population)):
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict

                with open(save_dir / "evolve_population.yaml", "w") as outfile:
                    yaml.dump(save_dict, outfile, default_flow_style=False)

            # Adaptive elite size
            # 自适应精英的大小
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))
            # Evaluate the fitness of each individual in the population
            # 平菇种群中每个个体的适应度
            fitness_scores = []
            for individual in population:
                for key, value in zip(hyp_GA.keys(), individual):
                    hyp_GA[key] = value
                hyp.update(hyp_GA)
                results = train(hyp.copy(), opt, device, callbacks)
                callbacks = Callbacks()
                # Write mutation results
                # 写入变异结果
                keys = (
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                )
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                fitness_scores.append(results[2])

            # Select the fittest individuals for reproduction using adaptive tournament selection
            # 使用"自适应锦标赛选择"选择适应度最高的进行繁殖
            selected_indices = []
            for _ in range(pop_size - elite_size):
                # Adaptive tournament size
                # 自适应
                tournament_size = max(
                    max(2, tournament_size_min),
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
                )
                # Perform tournament selection to choose the best individual
                # 执行锦标赛选择从而挑选出最佳的个体
                tournament_indices = random.sample(range(pop_size), tournament_size)
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                selected_indices.append(winner_index)

            # Add the elite individuals to the selected indices
            # 将精英个体添加到选定的索引中
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
            selected_indices.extend(elite_indices)
            # Create the next generation through crossover and mutation
            # 通过交叉和变异创造下一代
            next_generation = []
            for _ in range(pop_size):
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]
                # Adaptive crossover rate
                # 自适应交叉(交配)比例
                crossover_rate = max(
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
                )
                if random.uniform(0, 1) < crossover_rate:
                    crossover_point = random.randint(1, len(hyp_GA) - 1)
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
                else:
                    child = population[parent1_index]
                # Adaptive mutation rate
                # 自适应变异比例
                mutation_rate = max(
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
                )
                for j in range(len(hyp_GA)):
                    if random.uniform(0, 1) < mutation_rate:
                        child[j] += random.uniform(-0.1, 0.1)
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
                next_generation.append(child)
            # Replace the old population with the new generation
            # 用新一代替代旧种群
            population = next_generation
        # Print the best solution found
        # 打印找到的最佳解决方案
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]
        print("Best solution found:", best_individual)
        # Plot results 将结果可视化 / 输出保存信息
        plot_evolve(evolve_csv)
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Usage example: $ python train.py --hyp {evolve_yaml}'
        )


"""
生成一个随机的浮点数，范围在-10到10之间
"""


def generate_individual(input_ranges, individual_length):
    """
    Generate a random individual with gene values within specified input ranges.

    Args:
        input_ranges (list[tuple[float, float]]): List of tuples where each tuple contains the lower and upper bounds
            for the corresponding gene.
        individual_length (int): The number of genes in the individual.

    Returns:
        list[float]: A list representing a generated individual with random gene values within the specified ranges.

    Examples:
        ```python
        input_ranges = [(0.01, 0.1), (0.1, 1.0), (0.9, 2.0)]
        individual_length = 3
        individual = generate_individual(input_ranges, individual_length)
        print(individual)  # Output: [0.035, 0.678, 1.456] (example output)
        ```
    """
    individual = []
    for i in range(individual_length):
        lower_bound, upper_bound = input_ranges[i]
        individual.append(random.uniform(lower_bound, upper_bound))
    return individual


'''===============================五、run（）函数=========================================='''


def run(**kwargs):
    """
    Executes YOLOv5 training with given options, allowing optional overrides through keyword arguments.

    Args:
        weights (str, optional): Path to initial weights. Defaults to ROOT / 'yolov5s.pt'.
        cfg (str, optional): Path to model YAML configuration. Defaults to an empty string.
        data (str, optional): Path to dataset YAML configuration. Defaults to ROOT / 'data/coco128.yaml'.
        hyp (str, optional): Path to hyperparameters YAML configuration. Defaults to ROOT / 'data/hyps/hyp.scratch-low.yaml'.
        epochs (int, optional): Total number of training epochs. Defaults to 100.
        batch_size (int, optional): Total batch size for all GPUs. Use -1 for automatic batch size determination. Defaults to 16.
        imgsz (int, optional): Image size (pixels) for training and validation. Defaults to 640.
        rect (bool, optional): Use rectangular training. Defaults to False.
        resume (bool | str, optional): Resume most recent training with an optional path. Defaults to False.
        nosave (bool, optional): Only save the final checkpoint. Defaults to False.
        noval (bool, optional): Only validate at the final epoch. Defaults to False.
        noautoanchor (bool, optional): Disable AutoAnchor. Defaults to False.
        noplots (bool, optional): Do not save plot files. Defaults to False.
        evolve (int, optional): Evolve hyperparameters for a specified number of generations. Use 300 if provided without a value.
        evolve_population (str, optional): Directory for loading population during evolution. Defaults to ROOT / 'data/hyps'.
        resume_evolve (str, optional): Resume hyperparameter evolution from the last generation. Defaults to None.
        bucket (str, optional): gsutil bucket for saving checkpoints. Defaults to an empty string.
        cache (str, optional): Cache image data in 'ram' or 'disk'. Defaults to None.
        image_weights (bool, optional): Use weighted image selection for training. Defaults to False.
        device (str, optional): CUDA device identifier, e.g., '0', '0,1,2,3', or 'cpu'. Defaults to an empty string.
        multi_scale (bool, optional): Use multi-scale training, varying image size by ±50%. Defaults to False.
        single_cls (bool, optional): Train with multi-class data as single-class. Defaults to False.
        optimizer (str, optional): Optimizer type, choices are ['SGD', 'Adam', 'AdamW']. Defaults to 'SGD'.
        sync_bn (bool, optional): Use synchronized BatchNorm, only available in DDP mode. Defaults to False.
        workers (int, optional): Maximum dataloader workers per rank in DDP mode. Defaults to 8.
        project (str, optional): Directory for saving training runs. Defaults to ROOT / 'runs/train'.
        name (str, optional): Name for saving the training run. Defaults to 'exp'.
        exist_ok (bool, optional): Allow existing project/name without incrementing. Defaults to False.
        quad (bool, optional): Use quad dataloader. Defaults to False.
        cos_lr (bool, optional): Use cosine learning rate scheduler. Defaults to False.
        label_smoothing (float, optional): Label smoothing epsilon value. Defaults to 0.0.
        patience (int, optional): Patience for early stopping, measured in epochs without improvement. Defaults to 100.
        freeze (list, optional): Layers to freeze, e.g., backbone=10, first 3 layers = [0, 1, 2]. Defaults to [0].
        save_period (int, optional): Frequency in epochs to save checkpoints. Disabled if < 1. Defaults to -1.
        seed (int, optional): Global training random seed. Defaults to 0.
        local_rank (int, optional): Automatic DDP Multi-GPU argument. Do not modify. Defaults to -1.

    Returns:
        None: The function initiates YOLOv5 training or hyperparameter evolution based on the provided options.

    Examples:
        ```python
        import train
        train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
        ```

    Notes:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    # 这段代码主要是使得支持指令执行这个脚本

    # 执行这个脚本/ 调用train函数 / 开启训练
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        # setattr() 赋值属性，属性不存在则创建一个赋值
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
