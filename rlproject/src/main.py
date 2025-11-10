import traceback
from stable_baselines3 import PPO,SAC
import numpy as np
import cv2
import time
import mediapipe as mp
from custom_env import CustomEnv,EnvironmentRenderer
import pygame
import rtde_receive
import rtde_control
from cv.get_workspace import get_workspace
from collections import deque

trajectory_robot = deque(maxlen=60)
trajectory = deque(maxlen=60)

from camera_calibration.camera_calibration import CameraCalibration
from robot_control.ur_control import URControl  
from cv.hand_detect import HandDetection

from ultralytics import YOLO
cv_model = YOLO('runs/detect/train3/weights/best.onnx')
hand_detector = HandDetection()
cali = CameraCalibration()
robot_ip = "192.168.1.2"

robot_control = URControl(robot_ip)


env = CustomEnv()
env.random = False

w_env, h_env = 20, 10  # 环境的宽度和高度
model = SAC.load(
    "C:/Users/admin/Desktop/huifeng/RL/src/logs/best_model_sac59/best_model.zip",
    env=env,
    custom_objects={
        "observation_space": env.observation_space,
        "action_space": env.action_space
    }
)

env.stride_robot=2


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



desired_width = 2592 
desired_height = 1944





cap = cv2.VideoCapture(0)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)  # 创建一个窗口来显示矫正后的图像
cv2.namedWindow('edges', cv2.WINDOW_NORMAL)  
if not cap.isOpened():
    print("Error: Could not open camera for demonstration.")
    exit()

clicked_x, clicked_y = 1000, 1000
position_hand_env = [0,0]
object_points = [] # 用于存储世界坐标系中的点
img_points = []  # 用于存储图像坐标系中的点




def mouse_callback(event, x, y, flags, param):
    """
    鼠标事件发生时被调用的函数。
    """

    global clicked_x, clicked_y # 声明使用全局变量
    # global position_hand_env
    if event == cv2.EVENT_LBUTTONDOWN: # 检测鼠标左键按下事件
        clicked_x, clicked_y = x, y
        # print(np.linalg.inv(H) @ np.array([x, y, 1],dtype=np.float32))
        # position_hand_env = np.linalg.inv(H) @ np.array([x, y, 1],dtype=np.float32)[:2]
        # print(np.linalg.inv(H) @ np.array([x, y, 1],dtype=np.float32))

        # position_hand_env = 

        return clicked_x, clicked_y

cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

frame_count = 0
fps = 0.0
fps_interval = 1.0
fps_ts = time.time()

last_trigger_time = time.time()
freq = 2

last_action = [0,0]

recorded_data = []
last_hand = 0
render =  EnvironmentRenderer(grid_size=10, cell_size=50)
try:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera for demonstration.")
            break
        # 使用 cv2.undistort 对每一帧进行畸变矫正
        undistorted_frame = cali.undistort_frame(frame)
        undistorted_frame = get_workspace(undistorted_frame)
        img_gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)



        h , w = undistorted_frame.shape[:2]


        # results = cv_model.predict(undistorted_frame, conf=0.5, save=False,imgsz=undistorted_frame.shape[1::-1])
        results = cv_model.predict(undistorted_frame, conf=0.7, save=False,imgsz=640,verbose=False)



        undistorted_frame, hand_positions = hand_detector.process_frame(undistorted_frame)
        for i, r in enumerate(results):
            boxes = r.boxes
            for box in boxes:
                # 提取边界框坐标

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x2-x1 > 100:
                    continue
                # 计算中心点
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # 在图像上绘制边界框和中心点
                trajectory_robot.append([int(cx), int(cy)])

                cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (255, 0, 255), 6)
                # cv2.circle(undistorted_frame, (cx, cy), 5, (255, 0, 0), -1)
                # cv2.putText(undistorted_frame, f'({cx}, {cy})', (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # print(f"Detected object at center: ({cx}, {cy})")

        
        if len(trajectory_robot) >= 2:
            i = 2
            for j in range(1, len(trajectory_robot)):
                cv2.line(undistorted_frame, trajectory_robot[j - 1], trajectory_robot[j], (0, 255, 255), int(i//2))
                i+=0.2
 

        if hand_positions:
            position_hand_env = hand_positions[0]/np.array([w/h_env,h/w_env])
            position_hand_env = position_hand_env[1],h_env - position_hand_env[0]









        # edges = cv2.Canny(img_gray, 100, 200)

        # ys, xs = np.where(edges > 0)
        # # 例如右边伸出
        # idx = np.argmin(xs)
        # tip = (xs[idx], ys[idx])
        # cv2.circle(undistorted_frame, tip, 100, (0, 0, 255), -1)

        hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)

        # 设置肤色范围（适用于常见肤色，可根据光照微调）
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)

        # 根据阈值生成mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        ys, xs = np.where(mask > 0)
        # 例如右边伸出
        try:
            idx = np.argmin(xs)
            tip = (xs[idx], ys[idx])
            cv2.circle(undistorted_frame, tip, 10, (0, 0, 255), -1)
            fixed_point = [tip[1]*20/h,10]
        except:
            fixed_point = [10,10]

        
        key = cv2.waitKey(1) 
        if key== ord('q'):
            break
        
        now = time.time()
        
        
        if now - last_trigger_time > 1/freq:


            *position_robot_world,z,rx,ry,rz = robot_control.get_robot_pose()  # 使用更新后的类名




            position_robot_pixel = cali.world_to_pixel(position_robot_world)

            # print(position_robot_pixel)
            
            position_robot_env =  position_robot_pixel[1]*w_env / h, h_env - position_robot_pixel[0]*h_env /w 





            robot = np.array([position_robot_env],dtype=np.float32)
            hand = np.array([position_hand_env],dtype=np.float32)
            stride_hand = np.linalg.norm(hand - last_hand)
            print(stride_hand)
            distance_to_object = np.linalg.norm(robot - hand)
            distance = np.array([distance_to_object],dtype=np.float32)
            boundary = np.array([min(position_robot_env[0],position_robot_env[1],10-position_robot_env[0],10-position_robot_env[1])],dtype=np.float32)
            last_action = np.array([last_action],dtype=np.float32)
            dist_arm = env.dist_point_to_segment_correct(robot.flatten(),hand.flatten(),[15,10])[0]
            # fixed_point = [tip[1]*20/h,10]

            stride_robot = env.stride_robot
            # stride_hand = 1
            obs = np.concatenate((robot.flatten(),hand.flatten(),last_action.flatten(),distance.flatten(),boundary.flatten(),np.array([dist_arm],dtype=np.float32),np.array(fixed_point,dtype=np.float32),np.array([stride_robot]),np.array([stride_hand])))
            
            action, _states = model.predict(obs, deterministic=True)
            last_action = action    
            print(f"obs:{obs}\n action:{action}")
            action_pixel = action * np.array([h/w_env,w/h_env])*stride_robot
            action_pixel = -action_pixel[1],action_pixel[0]
            # print(action_pixel)
            cv2.circle(undistorted_frame, (int(position_robot_pixel[0]), int(position_robot_pixel[1])), 10, (0, 255, 0), -1)
            # position_robot_pixel += np.array([action_pixel[0],action_pixel[1]])
            # position_robot_world = cali.pixel_to_world(position_robot_pixel)
            

            print(f"z:{z}")
            rx,ry,rz = 0.256,-0.229,-4.984
            # dist = np.linalg.norm(position_robot_pixel-[cx,cy])
            # cv2.putText(undistorted_frame, str(dist), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            # if  dist > 140:
            #     print(position_robot_pixel,[cx,cy])
                
            #     # cv2.circle(undistorted_frame, (cx, cy), 10, (255, 0, 0), -1)
            #     # cv2.circle(undistorted_frame, (int(position_robot_pixel[0]), int(position_robot_pixel[1])), 10, (0, 255, 0), -1)
            #     x,y = cali.pixel_to_world([cx,cy])
            #     # robot_control.move_robot([position_robot_world[0],position_robot_world[1],0.1319,rx,ry,rz],1/freq)
            #     # print(f"move to ({x:.2f},{y:.2f})")
            #     robot_control.move_robot([x,y,0.1307,rx,ry,rz],1/freq)

            # else:
            #     position_robot_pixel += np.array([action_pixel[0],action_pixel[1]])
            #     position_robot_world = cali.pixel_to_world(position_robot_pixel)
            #     robot_control.move_robot([position_robot_world[0],position_robot_world[1],0.1307,rx,ry,rz],1/freq)
            

            position_robot_pixel += np.array([action_pixel[0],action_pixel[1]])
            position_robot_world = cali.pixel_to_world(position_robot_pixel)
            robot_control.move_robot([position_robot_world[0],position_robot_world[1],0.125,rx,ry,rz],1/freq)
            

            

            last_hand = hand
            last_trigger_time = now
            trajectory.append(position_robot_env)
            render.render(obs[:2],obs[2:4],fixed_point,trajectory)

        cv2.setMouseCallback("Frame", mouse_callback)
        frame_count += 1
        
        if (time.time() - fps_ts) > fps_interval:
            fps = frame_count / (time.time() - fps_ts)
            frame_count = 0
            fps_ts = time.time()
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.putText(undistorted_frame, f"FPS: {int(fps):d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Frame', undistorted_frame)
        cv2.imshow('edges', mask)


except Exception as e:
    print(f"Error occurred: {e}")
    traceback.print_exc()
    

finally:
    robot_control.disconnect()
    hand_detector.release()
    cap.release()
    cv2.destroyAllWindows()
    pygame.display.quit()
    pygame.quit()
