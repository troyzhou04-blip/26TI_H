# 要求346 整合版：上电 -> START 触摸按键 -> 选择功能（要求3/要求4/要求6）-> 倒计时
# -> 平台按所选功能的 TRIM 转到平衡位置（水平基准）-> 按 OK 确认 -> 运行所选功能。
# 三个功能共用检测/滤波/轨迹预测框架，各自使用自己的控制参数（互不影响，调一个不动其他）。
#   要求3：航点序列平衡（中位放稳 -> 滚到 dx=+160 -> 折返 -160 -> 在 -160 停稳），TRIM 同要求3.py
#   要求4：固定中线（屏幕中央）平衡，中线 ±30 微调区柔和参数，TRIM 同要求4.py
#   要求6：动态中线平衡（球静止 START_HOLD 秒按当前位置锁中线）±30 微调区，TRIM 同要求4.py
# 选择功能倒计时结束后平台按所选功能的 TRIM 转到平衡位置，按 OK 确认后才进入主控程序；
# 等待按键/选择/倒计时/确认期间平台保持水平；无触摸屏时跳过按键与确认，直接按 DEFAULT_MODE 运行。
# 模型部署方式：K230 原生 SDK（deploy_config.json + nncase_runtime + aicube 后处理），
# 模型与后处理配置从 /sdcard/mp_deployment_source/deploy_config.json 读取。
# 舵机方向约定：正转 -> 小球向 dx 负侧滚动；反转 -> 向 dx 正侧滚动。
# 若实际动作相反，将 DIR 改为 -1 即可。

import os, sys, gc, time
import ujson
import aicube
import image
import ulab.numpy as np
import nncase_runtime as nn
from media.sensor import *
from media.display import *
from media.media import *
from machine import PWM, FPIOA, Pin, TOUCH

# ---------------- 模型部署配置（同 mp_deployment 示例） ----------------
root_path  = "/sdcard/mp_deployment_source/"
config_path = root_path + "deploy_config.json"

def read_deploy_config(config_path):
    # 打开JSON文件以进行读取deploy_config
    with open(config_path, 'r') as json_file:
        try:
            config = ujson.load(json_file)
        except ValueError as e:
            print("JSON 解析错误:", e)
    return config

deploy_conf = read_deploy_config(config_path)
kmodel_name = deploy_conf["kmodel_path"]          # kmodel 文件名（相对 root_path）
labels = deploy_conf["categories"]                # 类别名列表
confidence_threshold = deploy_conf["confidence_threshold"]
nms_threshold = deploy_conf["nms_threshold"]
img_size = deploy_conf["img_size"]                # 模型输入尺寸 [w, h]
num_classes = deploy_conf["num_classes"]
nms_option = deploy_conf["nms_option"]
model_type = deploy_conf["model_type"]
if model_type == "AnchorBaseDet":
    anchors = deploy_conf["anchors"][0] + deploy_conf["anchors"][1] + deploy_conf["anchors"][2]
strides = [8, 16, 32]

# 重叠检测框合并阈值（IoU > 此值则合并），同一个球被检出多个重叠框时合并为一个
MERGE_IOU_THRESH = 0.3

def merge_overlap_boxes(boxes):
    """合并重叠的检测框（IoU > 阈值则合并为一个外接框）"""
    if len(boxes) <= 1:
        return boxes
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        # 取当前框作为起始
        x1, y1, x2, y2 = boxes[i][2], boxes[i][3], boxes[i][4], boxes[i][5]
        best_conf = boxes[i][1]
        best_cls = boxes[i][0]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = boxes[j][2], boxes[j][3], boxes[j][4], boxes[j][5]
                # 计算IoU
                ix1 = max(x1, bx1); iy1 = max(y1, by1)
                ix2 = min(x2, bx2); iy2 = min(y2, by2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (x2 - x1) * (y2 - y1)
                area_b = (bx2 - bx1) * (by2 - by1)
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0
                if iou > MERGE_IOU_THRESH:
                    # 合并：扩大外接框
                    x1 = min(x1, bx1); y1 = min(y1, by1)
                    x2 = max(x2, bx2); y2 = max(y2, by2)
                    if boxes[j][1] > best_conf:
                        best_conf = boxes[j][1]
                        best_cls = boxes[j][0]
                    used[j] = True
                    changed = True
        merged.append([best_cls, best_conf, x1, y1, x2, y2])
    return merged

# ---------------- 显示/采集配置（同 mp_deployment 示例） ----------------
display_mode = "lcd"
if display_mode == "lcd":
    DISPLAY_WIDTH = ALIGN_UP(800, 16)
    DISPLAY_HEIGHT = 480
else:
    DISPLAY_WIDTH = ALIGN_UP(1920, 16)
    DISPLAY_HEIGHT = 1080

OUT_RGB888P_WIDTH = ALIGN_UP(1080, 16)   # AI 采集通道分辨率（chn2, RGB888P）
OUT_RGB888P_HEIGH = 720

GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
MID_X = 400  # 屏幕中间竖直线 x 坐标（DISPLAY 坐标系），要求3/4 的中线、要求6 锁定前的临时参考

# ---------------- 舵机配置（方式同 importnumpy.py） ----------------
SERVO_PIN   = 42        # GPIO42，排针物理引脚 9
PWM_CHANNEL = 0         # PWM0
PWM_FREQ    = 50        # 50Hz，周期 20000us
MIN_US      = 500       # 0°   对应脉宽
MAX_US      = 2500      # 270° 对应脉宽
MAX_ANGLE   = 270
CENTER      = MAX_ANGLE / 2   # 135°，中位（平台水平）

# ---------------- 共用控制参数（三个功能取值一致，同要求3/4/6.py 验证值，调参改这里） ----------------
KP            = 0.12    # 比例系数：度/像素
KD            = 0.1     # 微分系数：度/(像素/秒)，速度阻尼，光滑管道必须够大
KI            = 0.030   # 低速积分系数：度/(像素·秒)，消除稳态偏移（球停住偏一侧时缓慢修正）
I_MAX         = 8.0     # 积分项限幅（度），防 windup
V_STILL       = 30.0    # 判静止速度（像素/秒），球速低于该值才允许积分累积
CALIB         = False   # 校准模式：舵机固定在中位+trim不控制（所选功能的 TRIM），用于验证管子水平
MAX_OFFSET    = 24.0    # 平滑饱和的渐近上限（度），大偏差时实际倾角明显小于该值
U_MIN         = 5.0     # 最小有效倾角（度），克服静摩擦，让球在目标点附近也能启动
APPROACH_K    = 16.0    # 接近限速系数：允许速度 = APPROACH_K*sqrt(剩余距离) + V_STILL（像素/秒）
APPROACH_BRAKE= 0.05    # 超速刹车增益：度/(像素/秒)，超出允许速度部分转为刹车倾角
BRAKE_MAX     = 6.0     # 额外刹车倾角上限（度），不宜过大防止中途减速过多卡死
DEADBAND      = 3       # 像素死区，距目标点/中线小于该值视为已到达
EMA_ALPHA     = 0.35    # dx 一阶低通滤波系数（0~1，越小越平滑但响应越慢）
V_ALPHA       = 0.30    # 速度估计滤波系数（用于轨迹预测和微分项）
PREDICT_MAX   = 0.4     # 检测丢失后轨迹外推的最长时间（秒），超时回中
MAX_STEP      = 2.5     # 每帧舵机角度最大变化量（度），防止突跳冲击小球
DIR           = 1       # 方向：角度增大=正转(小球滚向 dx 负侧)。实际相反则改为 -1
LOST_RECENTER = 0.3     # 目标丢失时每帧回中速度（度）

# ============ 要求3 参数（航点序列，同要求3.py 验证值，只影响要求3） ============
R3_TRIM       = 4.1     # 平台机械零点手动微调（度），同要求3.py TRIM：管道平衡位置（水平基准）
STUCK_RATE    = 1.5     # 卡死增压速率（度/秒）：已倾斜但球不动时，每秒额外加大的倾角
STUCK_MAX     = 8.0     # 卡死增压上限（度）

# ---------------- 航点任务（要求3） ----------------
WAYPOINTS   = [160, -160]  # dx 目标序列：中位 -> +160 -> -160，最后停在 -160 保持平衡
REACH_TOL   = 15           # 到达判定容差（像素），需同时满足低速；放宽可防止球停在容差带外不触发折返
REACH_HOLD  = 0.0          # 在容差内保持该时长（秒）才折返；0=到达即折返不停留
START_TOL   = 25           # 启动判定范围：球在中位 ±该像素内静止才计时
R3_START_HOLD = 3.0        # 开局球在中位放稳该时长（秒）后自动开始执行序列

# ---- 停稳（最后航点 -160） ----
SETTLE_TOL       = 30    # 停稳判定范围（像素）：在 -160 ±该值内且静止时开始计时
SETTLE_HOLD      = 2.0   # 停稳保持时长（秒）：满后彻底不再调整

# ============ 要求3 最后一段（+160 -> -160）调参区 ============
# 最后一段行程 320 像素（第一段的两倍），球速明显更高，共用参数会在中途过早刹车、
# 球被反向倾角带回正侧（回弹）。本区参数只影响最后一段，第一段和要求4/6 不受牵连。

# ---- 滚动速度：嫌快调小 KP_SCALE / APPROACH_K；中途停下就调大 ----
R3_FINAL_KP_SCALE   = 0.35   # 比例增益缩放：决定滚动加速度，越小球滚得越慢
R3_FINAL_APPROACH_K = 11.0   # 允许速度系数：允许速度 = 该值*sqrt(剩余距离) + V_STILL
R3_FINAL_KD_SCALE   = 0.6    # 微分阻尼缩放：调大减速更狠，调小不易中途刹停

# ---- 刹车：超速时的反向修正 ----
R3_FINAL_BRAKE_SCALE= 0.4    # 刹车增益缩放：越大刹得越狠
R3_FINAL_BRAKE_MAX  = 2.8    # 刹车倾角上限（度）

# ---- 回正速度：球滚向 -160 时倾角收回的快慢 ----
R3_FINAL_LEVEL_STEP = 3.0    # 回正限速（度/帧）：调小=回正慢、球不易停半路但易过冲；调大反之

# ---- 过冲回修：冲过 -160 后的往回拉 ----
R3_FINAL_RETURN_SCALE= 0.08  # 回拉增益缩放：越小越柔和，太大易来回振荡
R3_FINAL_DEADBAND   = 10     # 死区（像素）：-160 ±该值内不纠偏，靠摩擦停稳
R3_RETURN_DELAY     = 0.4    # 回修延时（秒）：冲过后需停稳该时长才开始回拉，嫌回修过早就加大

# ============ 要求4 参数（固定中线平衡，同要求4.py 验证值，只影响要求4） ============
R4_KP_RIGHT_BOOST= 1.91    # 球在右侧（dx>0）时 KP 额外倍率：右侧回推更快（1.0=左右对称）
R4_MAX_OFFSET_RIGHT_BOOST = 1.25  # 球在右侧（dx>0）时 MAX_OFFSET 额外倍率：右侧允许更大倾角（1.0=左右对称）
R4_TRIM       = 3.9     # 平台机械零点手动微调（度），同要求4.py TRIM：管道平衡位置（水平基准）

# ---- 车启动信号（高低电平前馈，要求4） ----
# 车主控在启动加速期间输出有效电平，K230 读到后进入预倾顶住模式，
# 用固定倾角抵消惯性甩球，电平撤销后自动恢复正常闭环。
R4_BRACE_ENABLE = False    # 车启动前馈总开关：不需要此功能时改为 False，完全关闭
R4_BRACE_ANGLE  = 5.0      # 启动预倾角（度）：顶住惯性甩球方向的倾角，方向反了改成负值

# ---- 接球模式（扰动缓冲，无需外部信号，要求4） ----
# 车启动等瞬时大加速度会把球甩出异常高的速度。球速超过 DISTURB_V 判定为被甩，
# 进入接球模式：位置纠偏（P/I/刹车）全部冻结，只输出与球速相反的倾角
# 吸收球的动能；球速降回 DISTURB_EXIT_V 以下退出（回差防模式抖动）。
# 退出后进入恢复期：接球阻尼与正常 PID 输出按比例缓慢混合，避免瞬间切换猛打。
R4_DISTURB_ENABLE = True   # 接球模式总开关：不需要时改为 False
R4_DISTURB_V      = 100.0  # 扰动判定速度（像素/秒）：正常控制球速一般不超过 200，超过视为被甩
R4_DISTURB_EXIT_V = 70.0  # 退出接球模式速度（像素/秒）：低于该值进入恢复期
R4_DISTURB_KD     = 0.08   # 接球阻尼增益：度/(像素/秒)，输出与球速同号的倾角顶住球减速
R4_DISTURB_MAX    = 8.0   # 接球模式倾角上限（度）
R4_DISTURB_RIGHT_BOOST = 1.3  # 向右突动增益倍数：球突然向右冲时倾角略加强；不需要方向差异改为 1.0
R4_DISTURB_RECOVER   = 0.6    # 恢复时长（秒）：退出接球模式后阻尼权重从 1 缓慢降到 0

# ---- ±30 微调区（两条竖线内，要求4） ----
R4_FINE_ZONE        = 30     # 微调区（像素）：对应屏上两条竖线，区内启用柔和参数
R4_FINE_KP_SCALE    = 0.45   # 比例增益缩放：越小球滚得越慢越柔
R4_FINE_KD_SCALE    = 0.6    # 微分阻尼缩放：调大减速更狠，调小不易中途刹停
R4_FINE_DEADBAND    = 15     # 死区（像素）：中线 ±该值内不纠偏，靠摩擦停稳
R4_FINE_APPROACH_K  = 11.0   # 允许速度系数：允许速度 = 该值*sqrt(剩余距离) + V_STILL
R4_FINE_BRAKE_SCALE = 0.4    # 刹车增益缩放：越大刹得越狠
R4_FINE_BRAKE_MAX   = 2.8    # 刹车倾角上限（度）
R4_FINE_STEP        = 2.0    # 区内每帧舵机最大变化（度）：动作更缓慢

# ============ 要求6 参数（动态中线平衡，同要求6.py 验证值，只影响要求6） ============
R6_KP_RIGHT_BOOST= 1.91    # 球在右侧（dx>0）时 KP 额外倍率：右侧回推更快（1.0=左右对称）
R6_MAX_OFFSET_RIGHT_BOOST = 1.25  # 球在右侧（dx>0）时 MAX_OFFSET 额外倍率：右侧允许更大倾角（1.0=左右对称）
R6_TRIM       = 3.9     # 平台机械零点手动微调（度），同要求4.py TRIM：管道平衡位置（水平基准）

# ---- 动态中线锁定（要求6） ----
# 进入运行后不做闭环控制，平台保持水平；球被检测到且速度低于 V_STILL 时累计计时，
# 连续静止 R6_START_HOLD 秒后，按球当前（滤波后）位置锁定中线，±30 竖线随之中线移动；
# 球动了或检测丢失则计时清零重新等待。
R6_START_HOLD = 3.0        # 球静止该时长（秒）后锁定中线

# ---- 车启动信号（高低电平前馈，要求6） ----
R6_BRACE_ENABLE = False    # 车启动前馈总开关：不需要此功能时改为 False，完全关闭
R6_BRACE_ANGLE  = 5.0      # 启动预倾角（度）：顶住惯性甩球方向的倾角，方向反了改成负值

# ---- 接球模式（扰动缓冲，无需外部信号，要求6） ----
R6_DISTURB_ENABLE = True   # 接球模式总开关：不需要时改为 False
R6_DISTURB_V      = 100.0  # 扰动判定速度（像素/秒）
R6_DISTURB_EXIT_V = 70.0  # 退出接球模式速度（像素/秒）：低于该值进入恢复期
R6_DISTURB_KD     = 0.08   # 接球阻尼增益：度/(像素/秒)
R6_DISTURB_MAX    = 8.0   # 接球模式倾角上限（度）
R6_DISTURB_RIGHT_BOOST = 1.3  # 向右突动增益倍数：不需要方向差异改为 1.0
R6_DISTURB_RECOVER   = 0.6    # 恢复时长（秒）：退出接球模式后阻尼权重从 1 缓慢降到 0

# ---- ±30 微调区（两条竖线内，要求6） ----
R6_FINE_ZONE        = 30     # 微调区（像素）：对应屏上两条竖线，区内启用柔和参数
R6_FINE_KP_SCALE    = 0.45   # 比例增益缩放：越小球滚得越慢越柔
R6_FINE_KD_SCALE    = 0.6    # 微分阻尼缩放：调大减速更狠，调小不易中途刹停
R6_FINE_DEADBAND    = 15     # 死区（像素）：中线 ±该值内不纠偏，靠摩擦停稳
R6_FINE_APPROACH_K  = 11.0   # 允许速度系数：允许速度 = 该值*sqrt(剩余距离) + V_STILL
R6_FINE_BRAKE_SCALE = 0.4    # 刹车增益缩放：越大刹得越狠
R6_FINE_BRAKE_MAX   = 2.8    # 刹车倾角上限（度）
R6_FINE_STEP        = 2.0    # 区内每帧舵机最大变化（度）：动作更缓慢

# ---------------- 启动信号引脚配置（要求4/6 共用物理引脚） ----------------
START_PIN    = 5        # 启动信号 GPIO 号（按板子丝印选空闲引脚，注意信号须为 3.3V 电平并与车主控共地）
START_ACTIVE = 1        # 有效电平：1=高电平表示车正在启动加速，0=低电平有效

# ---------------- 触摸屏启动按键与功能选择（按需求6 扩展） ----------------
# 流程：START -> 选择功能（REQ3/REQ4/REQ6）-> 延迟 BTN_DELAY 秒 -> 平衡位置确认（OK）-> 运行。
# 等待按键/选择/倒计时期间平台保持水平、不做任何控制。
# 触摸屏初始化失败（无触摸版屏幕）时自动跳过按键与确认，直接运行 DEFAULT_MODE。
DEFAULT_MODE = 6            # 无触摸屏时默认运行的功能：3/4/6
BTN_DELAY    = 3.0          # 选择功能后延迟启动时间（秒）
BTN_X1       = 310          # START 按键左上角 x（DISPLAY 坐标系）
BTN_Y1       = 390          # START 按键左上角 y
BTN_W        = 180          # 按键宽
BTN_H        = 70           # 按键高
# 功能选择按键（REQ3/REQ4/REQ6 一行三个）
MB_Y         = 250          # 功能选择按键行 y 坐标
MB_W         = 180          # 功能选择按键宽
MB_H         = 70           # 功能选择按键高
MB_R3_X      = 60           # REQ3 按键左上角 x
MB_R4_X      = 310          # REQ4 按键左上角 x
MB_R6_X      = 560          # REQ6 按键左上角 x
# 注：CanMV v1.2.2 固件 TOUCH 不支持坐标旋转参数；
# 若实测触摸坐标与屏幕显示对不上（左右/上下颠倒），反馈后再加软件坐标映射

# ---------------- 平衡位置确认（OK 按键） ----------------
# 选择功能倒计时结束后，平台按所选功能的 TRIM 转到平衡位置（水平基准），
# 屏上显示 OK 按键，按 OK 确认后才进入主控程序；之后所有控制都以 CENTER + trim 为水平。
CB_Y         = 250          # 确认按键行 y 坐标（与 START 键错开，防误触）
CB_W         = 180          # 确认按键宽
CB_H         = 70           # 确认按键高
CB_OK_X      = 310          # OK（确认进入运行）按键左上角 x（居中）

# ---------------- 舵机初始化 ----------------
fpioa = FPIOA()
fpioa.set_function(SERVO_PIN, getattr(FPIOA, "PWM%d" % PWM_CHANNEL))
pwm = PWM(PWM_CHANNEL, PWM_FREQ, 0)
pwm.enable(True)

def set_angle(angle):
    """设置舵机角度（0~270）"""
    angle = max(0.0, min(MAX_ANGLE, angle))
    pulse_us = MIN_US + (MAX_US - MIN_US) * angle / MAX_ANGLE
    pwm.duty(pulse_us / 20000 * 100)

set_angle(CENTER)
time.sleep(1)

# ---------------- 启动信号引脚初始化 ----------------
if not (R4_BRACE_ENABLE or R6_BRACE_ENABLE):
    start_pin = None
elif START_ACTIVE == 1:
    start_pin = Pin(START_PIN, Pin.IN, Pin.PULL_DOWN)  # 高电平有效：默认下拉，无信号时为低
else:
    start_pin = Pin(START_PIN, Pin.IN, Pin.PULL_UP)    # 低电平有效：默认上拉，无信号时为高

# ---------------- 触摸屏初始化 ----------------
# 自动探测触摸设备号（0=固件默认，1=自定义设备），哪个能用用哪个
tp = None
for _dev in (0, 1):
    try:
        tp = TOUCH(_dev)    # CanMV v1.2.2 固件 TOUCH 只接受设备号一个参数
        print("TOUCH init ok, dev =", _dev)
        break
    except Exception as e:
        print("TOUCH(%d) init failed:" % _dev, e)
if tp is None:
    print("TOUCH unavailable: skip buttons, run mode %d directly" % DEFAULT_MODE)

# ---------------- KPU / AI2D 初始化（同 mp_deployment 示例） ----------------
kmodel_frame_size = img_size
frame_size = [OUT_RGB888P_WIDTH, OUT_RGB888P_HEIGH]

# 计算padding值（保持长宽比缩放后四周补灰边）
ori_w = OUT_RGB888P_WIDTH
ori_h = OUT_RGB888P_HEIGH
width = kmodel_frame_size[0]
height = kmodel_frame_size[1]
ratiow = float(width) / ori_w
ratioh = float(height) / ori_h
ratio = ratiow if ratiow < ratioh else ratioh
new_w = int(ratio * ori_w)
new_h = int(ratio * ori_h)
dw = float(width - new_w) / 2
dh = float(height - new_h) / 2
top = int(round(dh - 0.1))
bottom = int(round(dh + 0.1))
left = int(round(dw - 0.1))
right = int(round(dw + 0.1))

# init kpu and load kmodel
kpu = nn.kpu()
ai2d = nn.ai2d()
kpu.load_kmodel(root_path + kmodel_name)
ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
ai2d.set_pad_param(True, [0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
ai2d_builder = ai2d.build([1, 3, OUT_RGB888P_HEIGH, OUT_RGB888P_WIDTH], [1, 3, height, width])

# ---------------- 传感器与显示初始化 ----------------
sensor = Sensor()
sensor.reset()
# 镜像/翻转保持原方向约定（画面旋转180度），控制方向参数 DIR 才有意义
sensor.set_hmirror(True)
sensor.set_vflip(True)
# 通道0直接给到显示VO，格式为YUV420
sensor.set_framesize(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
sensor.set_pixformat(PIXEL_FORMAT_YUV_SEMIPLANAR_420)
# 通道2给到AI做算法处理，格式为RGB888
sensor.set_framesize(width=OUT_RGB888P_WIDTH, height=OUT_RGB888P_HEIGH, chn=CAM_CHN_ID_2)
sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)
# 绑定通道0的输出到vo
sensor_bind_info = sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)
Display.bind_layer(**sensor_bind_info, layer=Display.LAYER_VIDEO1)
if display_mode == "lcd":
    Display.init(Display.ST7701, to_ide=True)
else:
    Display.init(Display.LT9611, to_ide=True)

# 创建OSD图像
osd_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)

# media初始化并启动sensor
MediaManager.init()
sensor.run()

# AI 输入/输出张量预分配
rgb888p_img = None
ai2d_input_tensor = None
data = np.ones((1, 3, width, height), dtype=np.uint8)
ai2d_output_tensor = nn.from_numpy(data)

clock = time.clock()

# ---------------- 控制状态 ----------------
servo_angle = CENTER   # 当前舵机角度
dx_f        = None     # 滤波/外推后的 dx（None 表示无目标且不预测），要求6 锁定后相对动态中线
dx_prev     = None     # 上一帧 dx_f，用于速度估计
v_f         = 0.0      # 滤波后的球速估计（像素/秒），用于轨迹预测和微分项
lost_t      = 0.0      # 检测连续丢失时长（秒）
i_term      = 0.0      # 低速积分项（度），消除稳态偏移
disturb_mode = False   # 接球模式标志（要求4/6）：球被瞬时大加速度甩动时为 True，纯阻尼吸收动能
recover_t   = 0.0      # 恢复期剩余时长（秒，要求4/6）：退出接球模式后从 DISTURB_RECOVER 递减到 0
center_locked = False  # 动态中线是否已锁定（要求6）
center_x    = MID_X    # 动态中线 x 坐标（要求6，锁定前为屏幕中央，仅作临时参考）
hold_t      = 0.0      # 要求3 启动/到达判定计时；要求6 锁定前球静止计时（秒）
wp_index    = -1       # 要求3：-1=等待球在中位放稳；0..len(WAYPOINTS)-1=当前航点序号
stuck_t     = 0.0      # 要求3：卡死计时（秒），已倾斜但球不动的持续时长
settle_t    = 0.0      # 要求3：停稳计时（秒），在 -160 ±SETTLE_TOL 内静止的持续时长
return_wait_t = 0.0    # 要求3：过冲回修计时（秒），冲过 -160 后静止等待回修的时长
u_prev      = 0.0      # 要求3：上一帧控制输出（度），用于最后一段回正限速
mode        = DEFAULT_MODE  # 当前功能：3=航点序列 4=固定中线 6=动态中线
start_state = 0 if tp is not None else 4  # 0=等待START 1=选择功能 2=倒计时 3=平衡位置确认 4=运行
btn_t       = 0.0      # 选择功能后启动倒计时累计（秒）
trim        = R3_TRIM if DEFAULT_MODE == 3 else (R4_TRIM if DEFAULT_MODE == 4 else R6_TRIM)  # 当前功能的 TRIM（度），管道平衡位置（水平基准），选择功能时按所选功能更新
tp_x        = None     # 本帧触摸点 x（无触摸为 None），用于屏上十字光标调试
tp_y        = None     # 本帧触摸点 y
t_prev      = time.ticks_us()

try:
    while True:
        clock.tick()
        rgb888p_img = sensor.snapshot(chn=CAM_CHN_ID_2)
        if rgb888p_img.format() == image.RGBP888:
            # ---- AI 推理（同 mp_deployment 示例） ----
            ai2d_input = rgb888p_img.to_numpy_ref()
            ai2d_input_tensor = nn.from_numpy(ai2d_input)
            ai2d_builder.run(ai2d_input_tensor, ai2d_output_tensor)

            kpu.set_input_tensor(0, ai2d_output_tensor)
            kpu.run()
            results = []
            for i in range(kpu.outputs_size()):
                out_data = kpu.get_output_tensor(i)
                result = out_data.to_numpy()
                result = result.reshape((result.shape[0] * result.shape[1] * result.shape[2] * result.shape[3]))
                del out_data
                results.append(result)
            gc.collect()

            # 后处理（按 deploy_config.json 中的模型类型选择）
            if model_type == "AnchorBaseDet":
                det_boxes = aicube.anchorbasedet_post_process(results[0], results[1], results[2], kmodel_frame_size, frame_size, strides, num_classes, confidence_threshold, nms_threshold, anchors, nms_option)
            elif model_type == "GFLDet":
                det_boxes = aicube.gfldet_post_process(results[0], results[1], results[2], kmodel_frame_size, frame_size, strides, num_classes, confidence_threshold, nms_threshold, nms_option)
            else:
                det_boxes = aicube.anchorfreedet_post_process(results[0], results[1], results[2], kmodel_frame_size, frame_size, strides, num_classes, confidence_threshold, nms_threshold, nms_option)
            # 合并重叠的检测框：同一个球的多个重叠框合并为一个外接框
            det_boxes = merge_overlap_boxes(det_boxes)
            # 只保留置信度最高的一个框，其余丢弃
            if len(det_boxes) > 1:
                best_det = det_boxes[0]
                for b in det_boxes:
                    if b[1] > best_det[1]:
                        best_det = b
                det_boxes = [best_det]

            # 实际帧间隔（秒），PID 的时间基准，不能假设固定帧率
            t_now = time.ticks_us()
            dt = time.ticks_diff(t_now, t_prev) / 1000000
            t_prev = t_now
            if dt <= 0:
                dt = 0.001

            # 从检测结果中选最可能的目标（面积最大的框），并画出所有框。
            # 检测框坐标为 AI 坐标系（1088x720），换算到显示坐标系（800x480）再使用
            best_cx = None
            best_area = -1
            if det_boxes:
                for det in det_boxes:
                    x1 = int(det[2] * DISPLAY_WIDTH / OUT_RGB888P_WIDTH)
                    y1 = int(det[3] * DISPLAY_HEIGHT / OUT_RGB888P_HEIGH)
                    x2 = int(det[4] * DISPLAY_WIDTH / OUT_RGB888P_WIDTH)
                    y2 = int(det[5] * DISPLAY_HEIGHT / OUT_RGB888P_HEIGH)
                    w = x2 - x1
                    h = y2 - y1
                    if w * h > best_area:
                        best_area = w * h
                        best_cx = (x1 + x2) // 2

            # 要求6 锁定前 dx 相对屏幕中央 MID_X（只用于速度估计和静止判定）；
            # 要求6 锁定后 dx 相对动态中线 center_x；要求3/4 始终相对屏幕中央 MID_X
            ref_x = center_x if (mode == 6 and center_locked) else MID_X

            dx_raw = None      # 本帧检测方框中心的原始 dx（未滤波），None 表示未检测到
            if best_cx is not None:
                dx = best_cx - ref_x
                dx_raw = dx  # 本帧方框中心的原始 dx，要求4/6 死区判定以它为准

                # 一阶低通滤波，抑制检测抖动（首次捕获直接初始化）
                if dx_f is None:
                    dx_f = dx
                    dx_prev = dx
                    v_f = 0.0
                else:
                    dx_f = dx_f + EMA_ALPHA * (dx - dx_f)
                    # 速度估计并滤波，供轨迹预测和微分项使用
                    v_meas = (dx_f - dx_prev) / dt
                    v_f = v_f + V_ALPHA * (v_meas - v_f)
                    dx_prev = dx_f
                lost_t = 0.0
            elif dx_f is not None and lost_t < PREDICT_MAX:
                # 检测丢失：按估计速度匀速外推小球轨迹，补偿识别掉帧/帧率不足
                lost_t += dt
                dx_f = dx_f + v_f * dt
                dx_f = max(-450, min(450, dx_f))  # 限制外推不漂出画面
                dx_prev = dx_f                    # 保持连续性，防再捕获时速度尖峰
                v_f = v_f * 0.98                  # 外推期间速度缓慢衰减，防止长时间漂移
            else:
                # 丢失超过 PREDICT_MAX：放弃预测，缓慢回中
                dx_f = None
                dx_prev = None
                v_f = 0.0

            # ---- 触摸屏流程：START -> 选功能（并按所选功能设置 TRIM）-> 倒计时 -> 平衡位置确认（OK）-> 运行 ----
            # 运行前的状态统一读一次触摸，记录坐标供屏上十字光标显示
            pts = []
            if start_state < 4:
                pts = tp.read(1)
                if len(pts) > 0:
                    tp_x = pts[0].x
                    tp_y = pts[0].y
                else:
                    tp_x = None
                    tp_y = None
            if start_state == 0:
                # 等待点按 START
                for i in range(len(pts)):
                    if (BTN_X1 <= pts[i].x <= BTN_X1 + BTN_W
                            and BTN_Y1 <= pts[i].y <= BTN_Y1 + BTN_H):
                        start_state = 1
                        break
            elif start_state == 1:
                # 选择功能：REQ3 / REQ4 / REQ6，选定后按所选功能设置 TRIM 并进入启动倒计时
                for i in range(len(pts)):
                    px = pts[i].x
                    py = pts[i].y
                    if MB_Y <= py <= MB_Y + MB_H:
                        if MB_R3_X <= px <= MB_R3_X + MB_W:
                            mode = 3
                        elif MB_R4_X <= px <= MB_R4_X + MB_W:
                            mode = 4
                        elif MB_R6_X <= px <= MB_R6_X + MB_W:
                            mode = 6
                        else:
                            continue
                        trim = R3_TRIM if mode == 3 else (R4_TRIM if mode == 4 else R6_TRIM)
                        start_state = 2
                        btn_t = 0.0
                        break
            elif start_state == 2:
                # 选择功能后延迟 BTN_DELAY 秒进入平衡位置确认
                btn_t += dt
                if btn_t >= BTN_DELAY:
                    start_state = 3
            elif start_state == 3:
                # 平衡位置确认：平台已按所选功能的 TRIM 转到水平基准（start_state < 4 分支保持），
                # 点 OK 才进入所选功能的主控程序
                for i in range(len(pts)):
                    px = pts[i].x
                    py = pts[i].y
                    if (CB_OK_X <= px <= CB_OK_X + CB_W
                            and CB_Y <= py <= CB_Y + CB_H):
                        start_state = 4
                        break
                if start_state == 4:
                    # 进入运行：清空全部控制状态，按所选功能从头开始
                    i_term = 0.0
                    disturb_mode = False
                    recover_t = 0.0
                    hold_t = 0.0
                    wp_index = -1
                    stuck_t = 0.0
                    settle_t = 0.0
                    return_wait_t = 0.0
                    u_prev = 0.0
                    center_locked = False
                    center_x = MID_X

            # ---- 动态中线锁定（要求6）：球连续静止 R6_START_HOLD 秒后按当前位置锁定中线 ----
            if start_state == 4 and mode == 6 and not center_locked:
                if best_cx is not None and abs(v_f) < V_STILL:
                    hold_t += dt
                    if hold_t >= R6_START_HOLD:
                        # 锁定：滤波后球位作为新中线，dx_f/dx_prev 同步平移保持连续
                        shift = int(round(dx_f))
                        center_x = MID_X + shift
                        center_locked = True
                        dx_f -= shift
                        dx_prev -= shift
                        hold_t = 0.0
                else:
                    hold_t = 0.0

            # 本帧画线用的中线位置：要求6 锁定后为固定中线，锁定前跟随球位预览；
            # 要求3/4 固定为屏幕中央
            if mode == 6:
                line_x = center_x if center_locked else (MID_X + int(round(dx_f)) if dx_f is not None else MID_X)
            else:
                line_x = MID_X

            osd_img.clear()
            # 中线（黄）及参考竖线（青）按功能绘制
            osd_img.draw_line(line_x, 0, line_x, 480, color=YELLOW, thickness=2)
            if mode == 3:
                # 要求3：dx=±160 航点竖直线及坐标标注
                osd_img.draw_line(line_x + 160, 0, line_x + 160, 480, color=CYAN, thickness=2)
                osd_img.draw_line(line_x - 160, 0, line_x - 160, 480, color=CYAN, thickness=2)
                osd_img.draw_string_advanced(line_x + 166, 12, 24, "+160", color=CYAN)
                osd_img.draw_string_advanced(line_x - 226, 12, 24, "-160", color=CYAN)
            else:
                # 要求4/6：dx=±30 参考竖线及坐标标注（要求6 随中线移动）
                osd_img.draw_line(line_x + 30, 0, line_x + 30, 480, color=CYAN, thickness=2)
                osd_img.draw_line(line_x - 30, 0, line_x - 30, 480, color=CYAN, thickness=2)
                osd_img.draw_string_advanced(line_x + 36, 12, 24, "+30", color=CYAN)
                osd_img.draw_string_advanced(line_x - 96, 12, 24, "-30", color=CYAN)

            # 画出所有检测框及球到当前中线的连线
            if det_boxes:
                for det in det_boxes:
                    x1 = int(det[2] * DISPLAY_WIDTH / OUT_RGB888P_WIDTH)
                    y1 = int(det[3] * DISPLAY_HEIGHT / OUT_RGB888P_HEIGH)
                    x2 = int(det[4] * DISPLAY_WIDTH / OUT_RGB888P_WIDTH)
                    y2 = int(det[5] * DISPLAY_HEIGHT / OUT_RGB888P_HEIGH)
                    w = x2 - x1
                    h = y2 - y1
                    osd_img.draw_rectangle(x1, y1, w, h, color=RED, thickness=3)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    osd_img.draw_line(cx, cy, line_x, cy, color=GREEN, thickness=2)

            # 是否在微调区（两条竖线内，要求4/6）：决定本帧使用的参数组和步进限速
            if mode == 4:
                fine_zone = dx_f is not None and abs(dx_f) <= R4_FINE_ZONE
            elif mode == 6:
                fine_zone = center_locked and dx_f is not None and abs(dx_f) <= R6_FINE_ZONE
            else:
                fine_zone = False

            # 读取车启动信号：有效电平期间进入预倾顶住模式（各功能总开关关闭时恒为 False）
            if mode == 4:
                car_starting = R4_BRACE_ENABLE and start_pin is not None and start_pin.value() == START_ACTIVE
            elif mode == 6:
                car_starting = R6_BRACE_ENABLE and start_pin is not None and start_pin.value() == START_ACTIVE
            else:
                car_starting = False

            if CALIB:
                # 校准模式：舵机固定在中位+trim（初值0即机械中位），不做闭环控制。
                # 观察球在管子任意位置是否基本不动，验证舵机机械中位是否水平
                i_term = 0.0
                target = CENTER + trim
            elif start_state < 4:
                # 等待按键/选择功能/倒计时/平衡位置确认阶段：不做闭环控制，
                # 平台跟随所选功能的 trim 保持水平（确认阶段即平衡位置）
                i_term = 0.0
                disturb_mode = False
                recover_t = 0.0
                stuck_t = 0.0
                target = CENTER + trim
            elif mode == 3:
                # ============ 要求3：航点序列状态机：中位放稳 -> +160 -> -160 -> 停在 -160 ============
                if dx_f is not None:
                    if wp_index < 0:
                        sp = 0  # 等待阶段目标保持中位（该阶段不做闭环控制，仅计时判定启动）
                        # 球在中位 ±START_TOL 内且静止，累计 R3_START_HOLD 秒后启动序列
                        if abs(dx_f) <= START_TOL and abs(v_f) < V_STILL:
                            hold_t += dt
                            if hold_t >= R3_START_HOLD:
                                wp_index = 0
                                hold_t = 0.0
                        else:
                            hold_t = 0.0
                    else:
                        sp = WAYPOINTS[wp_index]
                        # 到达判定：必须是真实检测（非轨迹预测）、进入容差且低速保持 REACH_HOLD
                        if (best_cx is not None and wp_index < len(WAYPOINTS) - 1
                                and abs(dx_f - sp) <= REACH_TOL and abs(v_f) < V_STILL):
                            hold_t += dt
                            if hold_t >= REACH_HOLD:
                                wp_index += 1
                                hold_t = 0.0
                        else:
                            hold_t = 0.0

                    # 误差相对当前目标点计算，死区以目标点为中心（高精度定位关键）
                    err = dx_f - sp
                    # 停稳计时：在最后航点 -160 ±SETTLE_TOL 内且静止时累计，否则清零
                    if (wp_index == len(WAYPOINTS) - 1 and abs(err) <= SETTLE_TOL
                            and abs(v_f) < V_STILL):
                        settle_t += dt
                    else:
                        settle_t = 0.0

                    # 等待启动（wait）阶段：不做调整，e 置 0 让平台只回正到水平
                    # 停稳 SETTLE_HOLD 秒后：彻底不再调整（e=0 时积分/卡死/刹车/静摩擦补偿全部不生效，平台回水平）
                    if wp_index < 0 or settle_t >= SETTLE_HOLD:
                        e = 0.0
                    else:
                        # 最后一段用加宽死区，进入 -160 附近后不纠偏
                        db = R3_FINAL_DEADBAND if wp_index == len(WAYPOINTS) - 1 else DEADBAND
                        e = err if abs(err) > db else 0.0

                    # 最后一段冲过 -160（err<0）后：停稳 R3_RETURN_DELAY 秒才开始往回带，
                    # 且误差按比例缩小，防止过早/过猛拉回在目标点两侧来回振荡；
                    # 等待期间 e=0 平台水平，球靠摩擦停稳
                    return_side = wp_index == len(WAYPOINTS) - 1 and err < 0
                    if return_side and abs(v_f) < V_STILL:
                        return_wait_t += dt
                    else:
                        return_wait_t = 0.0
                    if return_side:
                        if return_wait_t >= R3_RETURN_DELAY:
                            e = e * R3_FINAL_RETURN_SCALE
                        else:
                            e = 0.0

                    # 低速积分：球基本静止且偏离目标点时缓慢累积推力，
                    # 消除机械零点偏差和静摩擦导致的稳态偏移；球动起来则泄漏衰减
                    if e != 0 and abs(v_f) < V_STILL:
                        i_term = i_term + KI * e * dt
                        i_term = max(-I_MAX, min(I_MAX, i_term))
                    else:
                        i_term = i_term * 0.98

                    # 微分阻尼加速度死区：低速时不刹车，
                    # 避免球缓慢爬向目标点时被 KD 提前刹停在死区外；等待阶段也不刹车
                    v_d = v_f if abs(v_f) > V_STILL and wp_index >= 0 else 0.0
                    # 最后一段阻尼减弱，防止中途把球刹停
                    if wp_index == len(WAYPOINTS) - 1:
                        v_d = v_d * R3_FINAL_KD_SCALE

                    # PID：比例按偏差、微分按滤波速度、积分按低速累积量
                    # 最后一段（+160 -> -160）比例增益缩放，减慢滚动速度
                    kp = KP * R3_FINAL_KP_SCALE if wp_index == len(WAYPOINTS) - 1 else KP
                    u = kp * e + KD * v_d + i_term

                    # 平滑饱和：大偏差时增益自动变软，限制大误差下的倾角，
                    # 防止球加速过快直接冲过目标点（等效限制最大滚动速度）
                    u = MAX_OFFSET * u / (MAX_OFFSET + abs(u))

                    # 卡死增压：有偏差且球持续不动（倾斜了但球不走）时，
                    # 随时间线性加大倾角，直到球开始滚动；球一动立即复位
                    # 最后一段过冲侧不启用，避免往回修正被额外加大
                    if e != 0 and abs(v_f) < V_STILL and not return_side:
                        stuck_t += dt
                        boost = STUCK_RATE * stuck_t
                        if boost > STUCK_MAX:
                            boost = STUCK_MAX
                        u = u + boost if e > 0 else u - boost
                    else:
                        stuck_t = 0.0

                    # 提前减速（柔和）：只在朝目标运动且明显超速时刹车，
                    # 允许速度随剩余距离减小，让球接近目标点时提前降速平稳滑入。
                    # 允许速度有 V_STILL 兜底、刹车增益和上限都较小，不会把球刹停在半路
                    if e != 0 and (v_f > 0) == (err < 0):
                        # 最后一段用更低的允许速度系数，且刹车更柔和（增益缩放+更低上限）
                        final_leg = wp_index == len(WAYPOINTS) - 1
                        ak = R3_FINAL_APPROACH_K if final_leg else APPROACH_K
                        v_allow = ak * (abs(err) ** 0.5) + V_STILL
                        over = abs(v_f) - v_allow
                        if over > 0:
                            brake = APPROACH_BRAKE * over
                            if final_leg:
                                brake *= R3_FINAL_BRAKE_SCALE
                            bmax = R3_FINAL_BRAKE_MAX if final_leg else BRAKE_MAX
                            if brake > bmax:
                                brake = bmax
                            # 刹车方向与球速相反（与 KD 阻尼同向）：顶住小球减速
                            u = u + brake if v_f > 0 else u - brake

                    # 静摩擦补偿：有有效误差且控制器在纠偏方向输出过小时，
                    # 强制给到最小倾角让球启动；刹车方向（u 与 e 反向）不强制
                    # 最后一段过冲侧不强制，往回修正只用柔和的比例+积分输出
                    if e != 0 and -U_MIN < u < U_MIN and (u == 0 or (u > 0) == (e > 0)) and not return_side:
                        u = U_MIN if e > 0 else -U_MIN

                    # 最后一段接近侧（err>0，球滚向 -160）回正限速：
                    # 倾角每帧最多收回 R3_FINAL_LEVEL_STEP 度，避免平台回正过快、
                    # 球在低速时被摩擦力耗停在 -160 之前
                    if wp_index == len(WAYPOINTS) - 1 and err > 0 and u < u_prev:
                        u = max(u, u_prev - R3_FINAL_LEVEL_STEP)
                    u_prev = u

                    target = CENTER + trim + DIR * u
                else:
                    # 无目标且不预测：缓慢回中，避免平台停在倾角极限
                    i_term = 0.0
                    stuck_t = 0.0
                    settle_t = 0.0
                    return_wait_t = 0.0
                    if servo_angle > CENTER:
                        target = max(CENTER + trim, servo_angle - LOST_RECENTER)
                    else:
                        target = min(CENTER + trim, servo_angle + LOST_RECENTER)
            elif mode == 4:
                # ============ 要求4：固定中线（屏幕中央）平衡，闭环逻辑与原要求4 一致 ============
                if dx_f is not None:
                    # 死区：以检测方框中心的原始 dx（未滤波）为准，
                    # 中心附近不再输出，避免舵机围着中位抖动；
                    # 检测丢失（轨迹预测中）时回退用滤波值判定。
                    # 微调区内死区加宽，±R4_FINE_DEADBAND 内不纠偏靠摩擦停稳
                    db_ref = dx_raw if dx_raw is not None else dx_f
                    db = R4_FINE_DEADBAND if fine_zone else DEADBAND
                    e = dx_f if abs(db_ref) > db else 0.0

                    # 低速积分：球基本静止且偏出死区时缓慢累积推力，
                    # 消除机械零点偏差和静摩擦导致的稳态偏移；球动起来则泄漏衰减
                    if e != 0 and abs(v_f) < V_STILL:
                        i_term = i_term + KI * e * dt
                        i_term = max(-I_MAX, min(I_MAX, i_term))
                    else:
                        i_term = i_term * 0.98

                    # 微分阻尼加速度死区：低速时不刹车，
                    # 避免球缓慢爬向中心时被 KD 提前刹停在死区外。
                    # 微调区内阻尼减弱，防止把球刹停在半路
                    v_d = v_f if abs(v_f) > V_STILL else 0.0
                    if fine_zone:
                        v_d = v_d * R4_FINE_KD_SCALE

                    # PID：比例按偏差、微分按滤波速度、积分按低速累积量。
                    # 微调区内比例增益缩放，滚动更慢更柔
                    kp = KP * R4_FINE_KP_SCALE if fine_zone else KP
                    # 右侧提速：球在右侧（dx_f>0）且不在微调区时，KP 乘上额外倍率，回推更快
                    if not fine_zone and dx_f > 0:
                        kp *= R4_KP_RIGHT_BOOST
                    u = kp * e + KD * v_d + i_term

                    # 平滑饱和：大偏差时增益自动变软，限制大误差下的倾角，
                    # 防止球在远端加速过快直接冲过中线（等效限制最大滚动速度）
                    # 右侧上限略宽松（同 KP_RIGHT_BOOST 逻辑），允许右边角度稍大
                    max_off = MAX_OFFSET * R4_MAX_OFFSET_RIGHT_BOOST if (not fine_zone and dx_f > 0) else MAX_OFFSET
                    u = max_off * u / (max_off + abs(u))

                    # 提前减速（柔和）：只在朝中线运动且明显超速时刹车，
                    # 允许速度随剩余距离减小，让球接近中线时提前降速平稳滑入。
                    # 微调区内用更低的允许速度系数、更柔和的刹车
                    if e != 0 and (v_f > 0) == (dx_f < 0):
                        ak = R4_FINE_APPROACH_K if fine_zone else APPROACH_K
                        v_allow = ak * (abs(dx_f) ** 0.5) + V_STILL
                        over = abs(v_f) - v_allow
                        if over > 0:
                            brake = APPROACH_BRAKE * over
                            if fine_zone:
                                brake *= R4_FINE_BRAKE_SCALE
                            bmax = R4_FINE_BRAKE_MAX if fine_zone else BRAKE_MAX
                            if brake > bmax:
                                brake = bmax
                            # 刹车方向与球速相反（与 KD 阻尼同向）：顶住小球减速
                            u = u + brake if v_f > 0 else u - brake

                    # 静摩擦补偿：有有效误差且控制器在纠偏方向输出过小时，
                    # 强制给到最小倾角让球启动；刹车方向（u 与 e 反向）不强制
                    if e != 0 and -U_MIN < u < U_MIN and (u == 0 or (u > 0) == (e > 0)):
                        u = U_MIN if e > 0 else -U_MIN

                    # 接球模式：球速异常大（被瞬时加速度甩动）时进入，纯速度阻尼吸收动能；
                    # 球速降回 R4_DISTURB_EXIT_V 以下退出（回差防抖动），位置纠偏期间全部冻结。
                    # 球突然向右冲（v_f>0）时增益乘 R4_DISTURB_RIGHT_BOOST，倾角略加强补偿突动。
                    if R4_DISTURB_ENABLE:
                        if not disturb_mode and abs(v_f) > R4_DISTURB_V:
                            disturb_mode = True
                        elif disturb_mode and abs(v_f) < R4_DISTURB_EXIT_V:
                            disturb_mode = False
                            recover_t = R4_DISTURB_RECOVER  # 开始缓慢恢复
                    else:
                        disturb_mode = False
                    if disturb_mode:
                        i_term = 0.0
                        recover_t = 0.0
                        # 阻尼方向：v_f>0（球向右冲）时 u>0 平台左倾顶住球，反之亦然
                        kd = R4_DISTURB_KD * R4_DISTURB_RIGHT_BOOST if v_f > 0 else R4_DISTURB_KD
                        u = kd * v_f
                        u = max(-R4_DISTURB_MAX, min(R4_DISTURB_MAX, u))
                    elif recover_t > 0:
                        # 恢复期：接球阻尼与正常 PID 输出线性混合，
                        # 阻尼权重从 1 随时间缓慢降到 0，避免退出瞬间猛打
                        recover_t -= dt
                        if recover_t < 0:
                            recover_t = 0.0
                        w = recover_t / R4_DISTURB_RECOVER
                        kd = R4_DISTURB_KD * R4_DISTURB_RIGHT_BOOST if v_f > 0 else R4_DISTURB_KD
                        u_d = kd * v_f
                        u_d = max(-R4_DISTURB_MAX, min(R4_DISTURB_MAX, u_d))
                        u = w * u_d + (1 - w) * u

                    # 车启动加速期间：覆盖为固定预倾角，顶住惯性甩球；
                    # 积分冻结，防止信号撤销瞬间控制器带着累积量猛打
                    if car_starting:
                        i_term = 0.0
                        u = R4_BRACE_ANGLE

                    target = CENTER + trim + DIR * u
                else:
                    # 无目标且不预测：缓慢回中，避免平台停在倾角极限
                    i_term = 0.0
                    disturb_mode = False
                    recover_t = 0.0
                    if servo_angle > CENTER:
                        target = max(CENTER + trim, servo_angle - LOST_RECENTER)
                    else:
                        target = min(CENTER + trim, servo_angle + LOST_RECENTER)
            else:
                # ============ 要求6：动态中线平衡，闭环逻辑与原要求6 一致 ============
                if not center_locked:
                    # 等待锁定阶段：不做闭环控制，平台保持水平
                    i_term = 0.0
                    disturb_mode = False
                    recover_t = 0.0
                    target = CENTER + trim
                elif dx_f is not None:
                    # 死区：以检测方框中心的原始 dx（未滤波）为准，
                    # 中心附近不再输出，避免舵机围着中位抖动；
                    # 检测丢失（轨迹预测中）时回退用滤波值判定。
                    # 微调区内死区加宽，±R6_FINE_DEADBAND 内不纠偏靠摩擦停稳
                    db_ref = dx_raw if dx_raw is not None else dx_f
                    db = R6_FINE_DEADBAND if fine_zone else DEADBAND
                    e = dx_f if abs(db_ref) > db else 0.0

                    # 低速积分：球基本静止且偏出死区时缓慢累积推力，
                    # 消除机械零点偏差和静摩擦导致的稳态偏移；球动起来则泄漏衰减
                    if e != 0 and abs(v_f) < V_STILL:
                        i_term = i_term + KI * e * dt
                        i_term = max(-I_MAX, min(I_MAX, i_term))
                    else:
                        i_term = i_term * 0.98

                    # 微分阻尼加速度死区：低速时不刹车，
                    # 避免球缓慢爬向中心时被 KD 提前刹停在死区外。
                    # 微调区内阻尼减弱，防止把球刹停在半路
                    v_d = v_f if abs(v_f) > V_STILL else 0.0
                    if fine_zone:
                        v_d = v_d * R6_FINE_KD_SCALE

                    # PID：比例按偏差、微分按滤波速度、积分按低速累积量。
                    # 微调区内比例增益缩放，滚动更慢更柔
                    kp = KP * R6_FINE_KP_SCALE if fine_zone else KP
                    # 右侧提速：球在右侧（dx_f>0）且不在微调区时，KP 乘上额外倍率，回推更快
                    if not fine_zone and dx_f > 0:
                        kp *= R6_KP_RIGHT_BOOST
                    u = kp * e + KD * v_d + i_term

                    # 平滑饱和：大偏差时增益自动变软，限制大误差下的倾角，
                    # 防止球在远端加速过快直接冲过中线（等效限制最大滚动速度）
                    # 右侧上限略宽松（同 KP_RIGHT_BOOST 逻辑），允许右边角度稍大
                    max_off = MAX_OFFSET * R6_MAX_OFFSET_RIGHT_BOOST if (not fine_zone and dx_f > 0) else MAX_OFFSET
                    u = max_off * u / (max_off + abs(u))

                    # 提前减速（柔和）：只在朝中线运动且明显超速时刹车，
                    # 允许速度随剩余距离减小，让球接近中线时提前降速平稳滑入。
                    # 微调区内用更低的允许速度系数、更柔和的刹车
                    if e != 0 and (v_f > 0) == (dx_f < 0):
                        ak = R6_FINE_APPROACH_K if fine_zone else APPROACH_K
                        v_allow = ak * (abs(dx_f) ** 0.5) + V_STILL
                        over = abs(v_f) - v_allow
                        if over > 0:
                            brake = APPROACH_BRAKE * over
                            if fine_zone:
                                brake *= R6_FINE_BRAKE_SCALE
                            bmax = R6_FINE_BRAKE_MAX if fine_zone else BRAKE_MAX
                            if brake > bmax:
                                brake = bmax
                            # 刹车方向与球速相反（与 KD 阻尼同向）：顶住小球减速
                            u = u + brake if v_f > 0 else u - brake

                    # 静摩擦补偿：有有效误差且控制器在纠偏方向输出过小时，
                    # 强制给到最小倾角让球启动；刹车方向（u 与 e 反向）不强制
                    if e != 0 and -U_MIN < u < U_MIN and (u == 0 or (u > 0) == (e > 0)):
                        u = U_MIN if e > 0 else -U_MIN

                    # 接球模式：球速异常大（被瞬时加速度甩动）时进入，纯速度阻尼吸收动能；
                    # 球速降回 R6_DISTURB_EXIT_V 以下退出（回差防抖动），位置纠偏期间全部冻结。
                    # 球突然向右冲（v_f>0）时增益乘 R6_DISTURB_RIGHT_BOOST，倾角略加强补偿突动。
                    if R6_DISTURB_ENABLE:
                        if not disturb_mode and abs(v_f) > R6_DISTURB_V:
                            disturb_mode = True
                        elif disturb_mode and abs(v_f) < R6_DISTURB_EXIT_V:
                            disturb_mode = False
                            recover_t = R6_DISTURB_RECOVER  # 开始缓慢恢复
                    else:
                        disturb_mode = False
                    if disturb_mode:
                        i_term = 0.0
                        recover_t = 0.0
                        # 阻尼方向：v_f>0（球向右冲）时 u>0 平台左倾顶住球，反之亦然
                        kd = R6_DISTURB_KD * R6_DISTURB_RIGHT_BOOST if v_f > 0 else R6_DISTURB_KD
                        u = kd * v_f
                        u = max(-R6_DISTURB_MAX, min(R6_DISTURB_MAX, u))
                    elif recover_t > 0:
                        # 恢复期：接球阻尼与正常 PID 输出线性混合，
                        # 阻尼权重从 1 随时间缓慢降到 0，避免退出瞬间猛打
                        recover_t -= dt
                        if recover_t < 0:
                            recover_t = 0.0
                        w = recover_t / R6_DISTURB_RECOVER
                        kd = R6_DISTURB_KD * R6_DISTURB_RIGHT_BOOST if v_f > 0 else R6_DISTURB_KD
                        u_d = kd * v_f
                        u_d = max(-R6_DISTURB_MAX, min(R6_DISTURB_MAX, u_d))
                        u = w * u_d + (1 - w) * u

                    # 车启动加速期间：覆盖为固定预倾角，顶住惯性甩球；
                    # 积分冻结，防止信号撤销瞬间控制器带着累积量猛打
                    if car_starting:
                        i_term = 0.0
                        u = R6_BRACE_ANGLE

                    target = CENTER + trim + DIR * u
                else:
                    # 无目标且不预测：缓慢回中，避免平台停在倾角极限
                    i_term = 0.0
                    disturb_mode = False
                    recover_t = 0.0
                    if servo_angle > CENTER:
                        target = max(CENTER + trim, servo_angle - LOST_RECENTER)
                    else:
                        target = min(CENTER + trim, servo_angle + LOST_RECENTER)

            # 步进限幅后输出到舵机；要求4/6 微调区内用更小的 FINE_STEP，动作更缓慢
            if mode == 4:
                max_step = R4_FINE_STEP if fine_zone else MAX_STEP
            elif mode == 6:
                max_step = R6_FINE_STEP if fine_zone else MAX_STEP
            else:
                max_step = MAX_STEP
            if target > servo_angle + max_step:
                servo_angle += max_step
            elif target < servo_angle - max_step:
                servo_angle -= max_step
            else:
                servo_angle = target
            set_angle(servo_angle)

            # 帧率与控制状态显示在屏幕上（不再 print，节省串口打印耗时）
            fps = clock.fps()
            info = "FPS: {:.1f}".format(fps)
            osd_img.draw_string_advanced(5, 12, 24, info, color=GREEN)
            if dx_f is not None:
                stat = "dx:{:+.1f} ang:{:.1f}".format(dx_f, servo_angle)
                osd_img.draw_string_advanced(5, 44, 24, stat, color=GREEN)
            # 当前功能显示在右上角
            osd_img.draw_string_advanced(650, 12, 24, "MODE: REQ%d" % mode, color=GREEN)
            if start_state == 0:
                # START 按键：绿色框+文字，点按区域见上面 BTN_ 配置
                osd_img.draw_rectangle(BTN_X1, BTN_Y1, BTN_W, BTN_H, color=GREEN, thickness=3)
                osd_img.draw_string_advanced(BTN_X1 + 30, BTN_Y1 + 18, 32, "START", color=GREEN)
                osd_img.draw_string_advanced(5, 76, 24, "PRESS START", color=GREEN)
            elif start_state == 1:
                # 功能选择按键：REQ3 / REQ4 / REQ6
                osd_img.draw_rectangle(MB_R3_X, MB_Y, MB_W, MB_H, color=CYAN, thickness=3)
                osd_img.draw_string_advanced(MB_R3_X + 45, MB_Y + 18, 32, "REQ3", color=CYAN)
                osd_img.draw_rectangle(MB_R4_X, MB_Y, MB_W, MB_H, color=CYAN, thickness=3)
                osd_img.draw_string_advanced(MB_R4_X + 45, MB_Y + 18, 32, "REQ4", color=CYAN)
                osd_img.draw_rectangle(MB_R6_X, MB_Y, MB_W, MB_H, color=CYAN, thickness=3)
                osd_img.draw_string_advanced(MB_R6_X + 45, MB_Y + 18, 32, "REQ6", color=CYAN)
                osd_img.draw_string_advanced(5, 76, 24, "SELECT MODE", color=CYAN)
            elif start_state == 2:
                # 选择功能后启动倒计时
                run_info = "RUN IN: {:.1f}s".format(BTN_DELAY - btn_t)
                osd_img.draw_string_advanced(5, 76, 24, run_info, color=YELLOW)
            elif start_state == 3:
                # 平衡位置确认界面：OK 按键 + 当前功能 TRIM（平台已按 TRIM 转到平衡位置）
                osd_img.draw_rectangle(CB_OK_X, CB_Y, CB_W, CB_H, color=GREEN, thickness=3)
                osd_img.draw_string_advanced(CB_OK_X + 70, CB_Y + 18, 32, "OK", color=GREEN)
                cal_info = "TRIM:{:+.1f} PRESS OK".format(trim)
                osd_img.draw_string_advanced(5, 76, 24, cal_info, color=YELLOW)
            elif mode == 3:
                # 要求3 运行状态：当前目标点
                if wp_index < 0:
                    sp_info = "sp: wait"
                else:
                    sp_info = "sp: {:+d}".format(WAYPOINTS[wp_index])
                osd_img.draw_string_advanced(5, 76, 24, sp_info, color=CYAN)
            elif mode == 6 and not center_locked:
                # 要求6 锁定倒计时：显示剩余静止时间
                wait_info = "HOLD: {:.1f}s".format(R6_START_HOLD - hold_t)
                osd_img.draw_string_advanced(5, 76, 24, wait_info, color=YELLOW)
            elif car_starting:
                osd_img.draw_string_advanced(5, 76, 24, "START!", color=RED)
            elif disturb_mode:
                osd_img.draw_string_advanced(5, 76, 24, "CATCH!", color=YELLOW)
            elif recover_t > 0:
                osd_img.draw_string_advanced(5, 76, 24, "RECOVER", color=CYAN)

            # 触摸点十字光标 + 坐标（仅运行前状态显示，用于确认按键对位）
            if start_state < 4 and tp_x is not None:
                tx = int(tp_x)
                ty = int(tp_y)
                osd_img.draw_line(tx - 15, ty, tx + 15, ty, color=RED, thickness=2)
                osd_img.draw_line(tx, ty - 15, tx, ty + 15, color=RED, thickness=2)
                osd_img.draw_string_advanced(5, 108, 24, "x:{:d} y:{:d}".format(tx, ty), color=RED)

            Display.show_image(osd_img, 0, 0, Display.LAYER_OSD3)
            gc.collect()
        rgb888p_img = None

except KeyboardInterrupt:
    pass
except Exception as e:
    print("error:", e)
finally:
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    # 退出前回中位，防止平台停在倾角上把钢珠甩出去
    set_angle(CENTER)
    time.sleep(0.5)
    pwm.enable(False)
    pwm.deinit()
    del ai2d_input_tensor
    del ai2d_output_tensor
    # 停止摄像头输出
    sensor.stop()
    # 去初始化显示设备
    Display.deinit()
    # 释放媒体缓冲区
    MediaManager.deinit()
    gc.collect()
    time.sleep(1)
    nn.shrink_memory_pool()
