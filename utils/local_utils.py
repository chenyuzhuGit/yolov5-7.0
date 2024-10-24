import datetime


def getLastDate():
    """
    格式化时间
    :return: 格式化的时间字符串，如：2024-10-24 22:26:10
    """
    # return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')


def get_screen_shot_img_name(splicing_content=''):
    """
    获取以时间字符串为主，后缀为辅，生成的截图名称
    :param splicing_content: 拼接内容
    :return: 时间+后缀，拼接的的名称字符串
    """
    time = getLastDate()
    return str(time + splicing_content)
