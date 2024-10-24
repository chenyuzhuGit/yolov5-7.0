"""
截图工具
"""
import pyautogui
import time
import win32gui

# 截屏位置（左上角起始坐标点，右下角坐标点）
position_record = None


def screen_shot(path='screenshot/detect/', img_name='default', img_suffix='.jpg'):
    """
    截图
    :param path:图片保存路径
    :param img_name:图片名称
    :param img_suffix:图片后缀
    :return: 返回图片路径名称
    """

    # 截取指定程序窗口图片
    screen_shot_obj = pyautogui.screenshot(region=position_record)
    # 保存图片
    img = screen_shot_obj.save(path + img_name + img_suffix)
    return img


def get_shot_position(handle_number):
    """
    获取截屏位置，并存到position_record变量中
    :param handle_number:句柄编号
    备注：
        1.先使用这个方法获取截图位置，并缓存在全局变量中
        2.因为截图位置基本上不会一直变化，所在单独拿出来，只执行一次就可以了
        3.后续循环调用直接循环screen_shot方法就可以了
    """
    if handle_number is None:
        raise Exception("must have handle number")

    # 编号转码
    hwnd = int(handle_number, 16)

    global position_record
    # 根据窗口句柄获取窗口位置和大小
    position_record = win32gui.GetWindowRect(hwnd)
    left, top, right, bottom = position_record
    position_record = (left, top, right - left, bottom - top)


# 测试
if __name__ == "__main__":
    handleNumber = ''
    # 获取截图位置
    get_shot_position(handleNumber)

    # 截图总数量
    total = 5
    # 起始数量
    length = 1

    while length < total:
        screen_shot('screenshot/test/')
        length += 1
        time.sleep(0.2)
