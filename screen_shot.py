'''
截图工具
'''
import win32gui
import pyautogui
import time

# 地下城的句柄为：000306EA

# 根据句柄截图
# 通过spy++工具，获取到句柄，这里需要将十六进制转为十进制
# C:\Users\chenyuzhu\Desktop\网课YOLOv5原理与源码解析\个人学习笔记\spy++
hwnd = int('002A07A4', 16)
# 根据窗口句柄获取窗口位置和大小
left, top, right, bottom = win32gui.GetWindowRect(hwnd)

# 截取窗口图片
length = 1
while length < 200:
    screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
    # 保存图片
    # screenshot.save("screenshot/" + str(length) + ".jpg")
    length += 1
    time.sleep(0.2)


# 获取屏幕大小
# screen_width, screen_height = pyautogui.size()
# 获取窗口句柄的回调函数
# def enum_windows_proc(hwnd, param):
#     if param in win32gui.GetWindowText(hwnd):
#         print(f'窗口句柄: {hwnd}')


# 要查找的窗口部分标题文本
# window_title_part = "Anaconda"

# 枚举所有窗口，查找包含指定文本的窗口
# win32gui.EnumWindows(enum_windows_proc, window_title_part)

# 根据窗口标题获取窗口句柄
# hwnd = win32gui.FindWindow(None, '地下城与勇士')
# print(hwnd)
# # 根据窗口类名获取窗口句柄
# hwnd = win32gui.FindWindow('EdgeWindowClass', None)
# print(hwnd)
# # 获取当前活动窗口句柄
# hwnd = win32gui.GetForegroundWindow()
# print(hwnd)



# strptime = datetime.strptime(time.localtime(), "%Y-%m-%dT%H:%M:%S.%fZ")
# print()