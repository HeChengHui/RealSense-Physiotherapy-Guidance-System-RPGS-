# Credit: https://github.com/IntelRealSense/librealsense/issues/1904#issuecomment-398434434
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
from sys import platform
import argparse
import json
import math
import csv
import serial
from serial import *

# Realsense initialisation ---------------------------------------------------------------------------------------------------------------------------------
# Configure depth and color streams
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)

# Start streaming
pipe_profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = pipe_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
# RealSense end initialisation -------------------------------------------------------------------------------------------------------------------------



# openpose initialisation ------------------------------------------------------------------------------------------------------------------------------
# Import Openpose
dir_path = os.path.dirname(os.path.realpath(__file__))

# Change these variables to point to the correct folder (Release/x64 etc.)
sys.path.append(dir_path + '/../../python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
import pyopenpose as op


# Flags
parser = argparse.ArgumentParser()
# parser.add_argument("--no_display", default=True, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# !!To use openpose in realsense, i need to be able to read image from a folder, then output the json to another folder,
# then realsense use that folder to import the json files
#
# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/" # DO NOT CHANGE
params["display"] = 0
params["render_pose"] = 0
params["number_people_max"] = 1
params["write_json"] = "Test_input_images_json" # I think this is still needed. Can try removing it.


# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Start openpose
opWrapper = op.WrapperPython(0) # DONT TOUCH THE VALUE
opWrapper.configure(params)
opWrapper.start()
# openpose end initialisation -----------------------------------------------------------------------------------------------------------------------------



# Guidance skeleton initialisation -------------------------------------------------------------------------------------------------------------------------
# END POSE
with open("D:/STUDIES/YEAR 3 - SEM 1/ESP3902/openpose-master/build/examples/tutorial_api_python/Pose Database/half_T_pose_end.json") as g:
    guidance_skele_json = json.loads(g.read())

guidance_skele_part_end = guidance_skele_json["part_candidates"]

#STARTING POSE
with open("D:/STUDIES/YEAR 3 - SEM 1/ESP3902/openpose-master/build/examples/tutorial_api_python/Pose Database/half_T_pose_start.json") as g:
    guidance_skele_json = json.loads(g.read())

guidance_skele_part_start = guidance_skele_json["part_candidates"]
# Guidance skeleton end initialisation ----------------------------------------------------------------------------------------------------------------------



# MATH initialisation for GUIDANCE angles ------------------------------------------------------------------------------------------------------------------------

with open(r"D:\STUDIES\YEAR 3 - SEM 1\ESP3902\openpose-master\build\examples\tutorial_api_python\Dev_Json\start.json") as start_json:
    start = json.load(start_json)
    for start_123 in start['start_123']:
        guidance_start_angle123 = start_123["angle"]
        start_angle123_upperAllowance = start_123["upper"]
        start_angle123_lowerAllowance = start_123["lower"]
    for start_234 in start['start_234']:
        guidance_start_angle234 = start_234["angle"]
        start_angle234_allowance = start_234["lower"]

with open(r"D:\STUDIES\YEAR 3 - SEM 1\ESP3902\openpose-master\build\examples\tutorial_api_python\Dev_Json\end.json") as end_json:
    end = json.load(end_json)
    for end_234 in end['end_234']:
        guidance_end_angle234 = end_234["angle"]
        end_angle234_allowance = end_234["lower"]
    for end_814 in end['end_814']:
        guidance_end_angle814 = end_814["angle"]
        end_angle814_upperAllowance = end_814["upper"]
        end_angle814_lowerAllowance = end_814["lower"]
    for end_14 in end['end_14']:
        guidance_planar_angle_chest_14 = end_14["angle"]
        planar_angle_chest_14_upperAllowance = end_14["upper"]
        planar_angle_chest_14_lowerAllowance = end_14["lower"]
    for end_13 in end['end_13']:
        guidance_planar_angle_chest_13 = end_13["angle"]
        planar_angle_chest_13_upperAllowance = end_13["upper"]
        planar_angle_chest_13_lowerAllowance = end_13["lower"]

with open(r"D:\STUDIES\YEAR 3 - SEM 1\ESP3902\openpose-master\build\examples\tutorial_api_python\Dev_Json\repetition.json") as rep_json:
    rep = json.load(rep_json)
    rep_value = rep['repetition'][0]['repetition_number']
# MATH end initialisation for GUIDANCE angles -----------------------------------------------------------------------------------------------------------------------



# Arduino read from serial initialisation --------------------------------------------------------------------------------------------------------------------------------
ser = serial.Serial('COM7', 9600)  # RMB TO CHANGE THE COM PORT BASE ON YOUR SETUP!
ser.xonxoff=1
ser.flushInput()
# Arduino read from serial initialisation end --------------------------------------------------------------------------------------------------------------------------------



# Drawing config ----------------------------------    *Realsense use BGR instead of RGB like a normal person. weirdo.
line_thickness = 2                               #|
colour_red = (0, 0, 255)                         #|
colour_green = (0, 128, 0)                       #|
colour_blue = (255,0,0)                          #|
font = cv2.FONT_HERSHEY_TRIPLEX                  #|
#--------------------------------------------------


# global variables
curr_frame = 0
curr_pose = 0 # 0 is start, 1 is end
csv_frames = 0 # for CSV only
curr_rot = 0 # 0 is starting rotation, 1 is ending rotation part one, 2 is ending rotation part two
counter = 0 # for how many cycles completed
help_switch = 0 # the help screen is not toggled
bol_start = True # for the counter. Only after started then can increment.


# Streaming loop
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()   # frames.get_depth_frame() is a 640x360 depth image
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x360 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # Intrinsics & Extrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # save the colour image into a folder in path
        path = "D:/STUDIES/YEAR 3 - SEM 1/ESP3902/openpose-master/build/examples/tutorial_api_python/Realsense_colour_images"
        cv2.imwrite(os.path.join(path , str(curr_frame) + '_colour_frame.png'), color_image)

        # OpenPose
        imagepath = r'D:\STUDIES\YEAR 3 - SEM 1\ESP3902\openpose-master\build\examples\tutorial_api_python\Realsense_colour_images\0_colour_frame.png'
        datum = op.Datum()
        imageToProcess = cv2.imread(imagepath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        np.userskele_json_part = datum.poseKeypoints
        # if(np.userskele_json_part.ndim > 0):   # !!IMPORTANT!! MAKE SURE THERE IS THIS CONDITION TO MAKE THE VARIABLE HAVE PROPER KEYPOINT
        #     print(np.userskele_json_part[0][0][0])
        # if(datum.poseKeypoints.ndim > 0):      # not having to copy the array over to a new variable works as well
        #     print(datum.poseKeypoints[0][4])


        # Stack both images horizontally, but now i only want the colour_image
        images = color_image

###########################################################################################################################################################################################################

        # user skeleton calculation  # NEW VERSION (DATUM)
        if(datum.poseKeypoints.ndim > 0):
            # 123
            if(datum.poseKeypoints[0][2].all and datum.poseKeypoints[0][1].all and datum.poseKeypoints[0][3].all):
                if(any(datum.poseKeypoints[0][2]) and any(datum.poseKeypoints[0][1]) and any(datum.poseKeypoints[0][3])):
                    # print("\n123")
                    AB = [datum.poseKeypoints[0][1][0] - datum.poseKeypoints[0][2][0], datum.poseKeypoints[0][1][1] - datum.poseKeypoints[0][2][1]]
                    AC = [datum.poseKeypoints[0][3][0] - datum.poseKeypoints[0][2][0], datum.poseKeypoints[0][3][1] - datum.poseKeypoints[0][2][1]]
                    # dot_prdt = np.array(AB) @ np.array(AC)
                    dot_prdt = np.dot(AB, AC)
                    # print("dot_prdt: ", dot_prdt)
                    len_AB = math.sqrt(AB[0]**2 + AB[1]**2)
                    len_AC = math.sqrt(AC[0]**2 + AC[1]**2)
                    Length = len_AB * len_AC
                    # print("Length: ", Length)
                    if(len_AB!=0 and len_AC!=0):
                        divided_value = np.divide(dot_prdt,Length)
                        # print("divided_value: ", divided_value)
                        if(divided_value<1 and divided_value>-1):
                            user_angle123 = math.degrees(math.acos(divided_value))
                            # print("user_angle123: ", user_angle123)
                        else:
                            user_angle123=0
                else:
                    user_angle123 = 0
            else:
                user_angle123 = 0 # needed else will not work. So in case when either 1 of the parts is null, then just let the angle =0

            # 234
            if(datum.poseKeypoints[0][2].all and datum.poseKeypoints[0][3].all and datum.poseKeypoints[0][4].all):
                if(any(datum.poseKeypoints[0][2]) and any(datum.poseKeypoints[0][3]) and any(datum.poseKeypoints[0][4])):
                    # print("\n234")
                    AB = [datum.poseKeypoints[0][2][0] - datum.poseKeypoints[0][3][0], datum.poseKeypoints[0][2][1] - datum.poseKeypoints[0][3][1]]
                    AC = [datum.poseKeypoints[0][4][0] - datum.poseKeypoints[0][3][0], datum.poseKeypoints[0][4][1] - datum.poseKeypoints[0][3][1]]
                    # dot_prdt = np.array(AB) @ np.array(AC)
                    dot_prdt = np.dot(AB, AC)
                    # print("dot_prdt: ", dot_prdt)
                    len_AB = math.sqrt(AB[0]**2 + AB[1]**2)
                    len_AC = math.sqrt(AC[0]**2 + AC[1]**2)
                    Length = len_AB * len_AC
                    # print("Length: ", Length)
                    if(len_AB!=0 and len_AC!=0):
                        divided_value = np.divide(dot_prdt,Length)
                        # print("divided_value: ", divided_value)
                        if(divided_value<1 and divided_value>-1):
                            # user_angle234 = math.degrees(math.acos(divided_value))
                            user_angle234 = math.degrees(math.acos(divided_value))
                            # print("user_angle234: ", user_angle234)
                        else:
                            user_angle234=0
                    else:
                        user_angle234=0

                    # print(user_angle234)
                else:
                    user_angle234 = 0
            else:
                user_angle234 = 0 # needed else will not work. So in case when either 1 of the parts is null, then just let the angle =0

            # 814
            if(datum.poseKeypoints[0][8].all and datum.poseKeypoints[0][1].all and datum.poseKeypoints[0][4].all):
                if(any(datum.poseKeypoints[0][2]) and any(datum.poseKeypoints[0][1]) and any(datum.poseKeypoints[0][4])):
                    # print("\n813")
                    AB = [datum.poseKeypoints[0][8][0] - datum.poseKeypoints[0][1][0], datum.poseKeypoints[0][8][1] - datum.poseKeypoints[0][1][1]]
                    AC = [datum.poseKeypoints[0][4][0] - datum.poseKeypoints[0][1][0], datum.poseKeypoints[0][4][1] - datum.poseKeypoints[0][1][1]]
                    # dot_prdt = np.array(AB) @ np.array(AC)
                    dot_prdt = np.dot(AB, AC)
                    # print("dot_prdt: ", dot_prdt)
                    len_AB = math.sqrt(AB[0]**2 + AB[1]**2)
                    len_AC = math.sqrt(AC[0]**2 + AC[1]**2)
                    Length = len_AB * len_AC
                    # print("Length: ", Length)
                    if(len_AB!=0 and len_AC!=0):
                        divided_value = np.divide(dot_prdt,Length)
                        # print("divided_value: ", divided_value)
                        if(divided_value<1 and divided_value>-1):
                            user_angle814 = math.degrees(math.acos(divided_value))
                            # print("user_angle814: ", user_angle814)
                        else:
                            user_angle814=0
                else:
                    user_angle814 = 0
            else:
                user_angle814 = 0 # needed else will not work. So in case when either 1 of the parts is null, then just let the angle =0

            # Normal of 1-9-12 @ 1-4 @ 1-3
            if(datum.poseKeypoints[0][1].all and datum.poseKeypoints[0][9].all and datum.poseKeypoints[0][12].all and datum.poseKeypoints[0][4].all and datum.poseKeypoints[0][3].all):
                if(any(datum.poseKeypoints[0][1]) and any(datum.poseKeypoints[0][9]) and any(datum.poseKeypoints[0][12]) and any(datum.poseKeypoints[0][4]) and any(datum.poseKeypoints[0][3])):
                    depth_keypoint1 = aligned_depth_frame.get_distance(int(datum.poseKeypoints[0][1][0]), int(datum.poseKeypoints[0][1][1])) #A
                    depth_keypoint9 = aligned_depth_frame.get_distance(int(datum.poseKeypoints[0][9][0]), int(datum.poseKeypoints[0][9][1])) #B
                    depth_keypoint12 = aligned_depth_frame.get_distance(int(datum.poseKeypoints[0][12][0]), int(datum.poseKeypoints[0][12][1])) #C
                    depth_keypoint3 = aligned_depth_frame.get_distance(int(datum.poseKeypoints[0][3][0]), int(datum.poseKeypoints[0][3][1])) #D
                    depth_keypoint4 = aligned_depth_frame.get_distance(int(datum.poseKeypoints[0][4][0]), int(datum.poseKeypoints[0][4][1])) #E

                    point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [int(datum.poseKeypoints[0][1][0]), int(datum.poseKeypoints[0][1][1])], depth_keypoint1)
                    point9 = rs.rs2_deproject_pixel_to_point(color_intrin, [int(datum.poseKeypoints[0][9][0]), int(datum.poseKeypoints[0][9][1])], depth_keypoint9)
                    point12 = rs.rs2_deproject_pixel_to_point(color_intrin, [int(datum.poseKeypoints[0][12][0]), int(datum.poseKeypoints[0][12][1])], depth_keypoint12)
                    point4 = rs.rs2_deproject_pixel_to_point(color_intrin, [int(datum.poseKeypoints[0][4][0]), int(datum.poseKeypoints[0][4][1])], depth_keypoint4)
                    point3 = rs.rs2_deproject_pixel_to_point(color_intrin, [int(datum.poseKeypoints[0][3][0]), int(datum.poseKeypoints[0][3][1])], depth_keypoint3)

                    len_AB = math.sqrt(math.pow(point1[0] - point9[0], 2) + math.pow(point1[1] - point9[1],2) + math.pow(point1[2] - point9[2], 2)) #1-9
                    len_AC = math.sqrt(math.pow(point1[0] - point12[0], 2) + math.pow(point1[1] - point12[1],2) + math.pow(point1[2] - point12[2], 2)) #1-12
                    len_AE = math.sqrt(math.pow(point1[0] - point4[0], 2) + math.pow(point1[1] - point4[1],2) + math.pow(point1[2] - point4[2], 2)) #1-4
                    len_AD = math.sqrt(math.pow(point1[0] - point3[0], 2) + math.pow(point1[1] - point3[1],2) + math.pow(point1[2] - point3[2], 2)) #1-3

                    AB = [point9[0] - point1[0], point9[1] - point1[1], depth_keypoint9-depth_keypoint1]
                    AC = [point12[0] - point1[0], point12[1] - point1[1], depth_keypoint12-depth_keypoint1]
                    AE = [point4[0] - point1[0], point4[1] - point1[1], depth_keypoint4-depth_keypoint1]
                    AD = [point3[0] - point1[0], point3[1] - point1[1], depth_keypoint3-depth_keypoint1]

                    if(len_AB != 0 and len_AC != 0 and len_AE != 0 and len_AD != 0):
                        cross_prdt = np.cross(AB, AC)
                        len_cross_prdt=math.sqrt(math.pow(cross_prdt[0], 2) + math.pow(cross_prdt[1], 2) + math.pow(cross_prdt[2], 2))
                        norm_cross_prdt = np.divide(cross_prdt, len_cross_prdt)
                        norm_dot_prdt = np.divide((np.array(norm_cross_prdt) @ np.array(AE)), (len_AE))
                        planar_angle_chest_14 = math.degrees(math.acos(norm_dot_prdt))
                        norm_dot_prdt = np.divide((np.array(norm_cross_prdt) @ np.array(AD)), (len_AD))
                        planar_angle_chest_13 = math.degrees(math.acos(norm_dot_prdt))

                        # print("planar: ", planar_angle_chest)

                    # print(planar_angle_chest)

                    # distance away from camera warning
                    # print(depth_keypoint1)
                    if(depth_keypoint1>3.0):
                        images = cv2.rectangle(images, (5, 340) ,(140, 360) , (255, 255, 255), -1)
                        cv2.putText(images,'Move forward',(7, 355),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                        bol_wrong = True
                    elif(depth_keypoint1<2.0):
                        # print("AJKLSHDLAJSDLKJASLKDJALKSDJLAKSJDLKAJSDLKAJSDLKJASLKDJALKSJDLKASJDLKAJSLKDJALSKJDALKSJDLKAJSDLKAJSDLKJASe")
                        images = cv2.rectangle(images, (5, 340) ,(140, 360) , (255, 255, 255), -1)
                        cv2.putText(images,"Move backward",(7, 355),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                        # print("asdasdasdasdasdasdasdasdasdasdas")
                        bol_wrong = True
                    else:
                        bol_wrong = False
        else:
            user_angle123 = 0
            user_angle234 = 0
            user_angle814 = 0
            planar_angle_chest_14 = 0
            planar_angle_chest_13 = 0

        # print("123:",user_angle123)
        # print("234:",user_angle234)
        # print("814:",user_angle814)
        # print("14:",planar_angle_chest_14)
        # print("13:",planar_angle_chest_13)


        # User rotation axis. 0-2 in terms of x,y,z
        ser_bytes = ser.readline()
        # print(ser_bytes)
        decoded_bytes = ser_bytes[0:len(ser_bytes)].decode("utf-8").strip('\n').split(',')
        # print(decoded_bytes)
        if"\x00" not in decoded_bytes[0] and decoded_bytes[0]!='':
        # if decoded_bytes[0].isdecimal() == True:
            float_list_values = [float(i) for i in decoded_bytes]
            highest_list_values = max(float_list_values, key=abs)
            highest_axis_index = float_list_values.index(highest_list_values)
            # print(highest_list_values)
            # print("rot index:",highest_axis_index)
        else:
            highest_axis_index = 2 # since we not using z-axis
            highest_list_values = 0 # doesnt matter what i put
        # problem with not reading the latest data from arduiono. it reads the top in a ever-filling container of data from arduino.
        # tried delay => fps worst
        # tried https://stackoverflow.com/questions/1093598/pyserial-how-to-read-the-last-line-sent-from-a-serial-device => doesnt work
        # tried https://stackoverflow.com/questions/1093598/pyserial-how-to-read-the-last-line-sent-from-a-serial-device => doesnt work


        # User skeleton drawing  # NEW VERSION (DATUM)
        if(datum.poseKeypoints.ndim > 0):
            for i in range(15):
                if(datum.poseKeypoints[0][i].all):  # need to .all to make sure that that keypoint returns true for it to be drawn
                    if(any(datum.poseKeypoints[0][i])):
                        cv2.circle(images, (int(datum.poseKeypoints[0][i][0]),int(datum.poseKeypoints[0][i][1])), 5, colour_blue, thickness=line_thickness)


        # Guidance skeleton drawing (Line version)
        if(curr_pose == 1): # change to end pose
            if((user_angle234<guidance_end_angle234-end_angle234_allowance) or (user_angle814<guidance_end_angle814-end_angle814_lowerAllowance or user_angle814>guidance_end_angle814+end_angle814_upperAllowance) or
                (planar_angle_chest_13<guidance_planar_angle_chest_13-planar_angle_chest_13_lowerAllowance or planar_angle_chest_13>guidance_planar_angle_chest_13+planar_angle_chest_13_upperAllowance) or
                (planar_angle_chest_14<guidance_planar_angle_chest_14-planar_angle_chest_14_lowerAllowance or planar_angle_chest_14>guidance_planar_angle_chest_14+planar_angle_chest_14_upperAllowance) or
                (curr_rot==1 and (highest_axis_index!=0 or highest_list_values<0)) or (curr_rot==2 and (highest_axis_index!=0 or highest_list_values>0)) or
                bol_wrong == True): # outside tolerance, red colour
                #0 - 1
                cv2.line(images, (int(guidance_skele_part_end[0]['0'][0]),int(guidance_skele_part_end[0]['0'][1])), (int(guidance_skele_part_end[0]['1'][0]),int(guidance_skele_part_end[0]['1'][1])), colour_red, thickness=line_thickness)
                #1 - 2
                cv2.line(images, (int(guidance_skele_part_end[0]['2'][0]),int(guidance_skele_part_end[0]['2'][1])), (int(guidance_skele_part_end[0]['1'][0]),int(guidance_skele_part_end[0]['1'][1])), colour_red, thickness=line_thickness)
                #1 - 5
                cv2.line(images, (int(guidance_skele_part_end[0]['5'][0]),int(guidance_skele_part_end[0]['5'][1])), (int(guidance_skele_part_end[0]['1'][0]),int(guidance_skele_part_end[0]['1'][1])), colour_red, thickness=line_thickness)
                #1 - 8
                cv2.line(images, (int(guidance_skele_part_end[0]['8'][0]),int(guidance_skele_part_end[0]['8'][1])), (int(guidance_skele_part_end[0]['1'][0]),int(guidance_skele_part_end[0]['1'][1])), colour_red, thickness=line_thickness)
                #2 - 3
                cv2.line(images, (int(guidance_skele_part_end[0]['2'][0]),int(guidance_skele_part_end[0]['2'][1])), (int(guidance_skele_part_end[0]['3'][0]),int(guidance_skele_part_end[0]['3'][1])), colour_red, thickness=line_thickness)
                #3 - 4
                cv2.line(images, (int(guidance_skele_part_end[0]['3'][0]),int(guidance_skele_part_end[0]['3'][1])), (int(guidance_skele_part_end[0]['4'][0]),int(guidance_skele_part_end[0]['4'][1])), colour_red, thickness=line_thickness)
                #5 - guidance_skele_part_end
                cv2.line(images, (int(guidance_skele_part_end[0]['5'][0]),int(guidance_skele_part_end[0]['5'][1])), (int(guidance_skele_part_end[0]['6'][0]),int(guidance_skele_part_end[0]['6'][1])), colour_red, thickness=line_thickness)
                #6 - 7
                cv2.line(images, (int(guidance_skele_part_end[0]['6'][0]),int(guidance_skele_part_end[0]['6'][1])), (int(guidance_skele_part_end[0]['7'][0]),int(guidance_skele_part_end[0]['7'][1])), colour_red, thickness=line_thickness)
                #8 - 9
                cv2.line(images, (int(guidance_skele_part_end[0]['8'][0]),int(guidance_skele_part_end[0]['8'][1])), (int(guidance_skele_part_end[0]['9'][0]),int(guidance_skele_part_end[0]['9'][1])), colour_red, thickness=line_thickness)
                #8 - 12
                cv2.line(images, (int(guidance_skele_part_end[0]['8'][0]),int(guidance_skele_part_end[0]['8'][1])), (int(guidance_skele_part_end[0]['12'][0]),int(guidance_skele_part_end[0]['12'][1])), colour_red, thickness=line_thickness)
                #9 - 10
                cv2.line(images, (int(guidance_skele_part_end[0]['9'][0]),int(guidance_skele_part_end[0]['9'][1])), (int(guidance_skele_part_end[0]['10'][0]),int(guidance_skele_part_end[0]['10'][1])), colour_red, thickness=line_thickness)
                #10 - 11
                cv2.line(images, (int(guidance_skele_part_end[0]['10'][0]),int(guidance_skele_part_end[0]['10'][1])), (int(guidance_skele_part_end[0]['11'][0]),int(guidance_skele_part_end[0]['11'][1])), colour_red, thickness=line_thickness)
                #12 - 13
                cv2.line(images, (int(guidance_skele_part_end[0]['12'][0]),int(guidance_skele_part_end[0]['12'][1])), (int(guidance_skele_part_end[0]['13'][0]),int(guidance_skele_part_end[0]['13'][1])), colour_red, thickness=line_thickness)
                #13 - 14
                cv2.line(images, (int(guidance_skele_part_end[0]['13'][0]),int(guidance_skele_part_end[0]['13'][1])), (int(guidance_skele_part_end[0]['14'][0]),int(guidance_skele_part_end[0]['14'][1])), colour_red, thickness=line_thickness)

                # Error warnings
                #rotation
                if(curr_rot==1 and (highest_axis_index!=0 or highest_list_values<0)):
                    # Draw a rectangle
                    images = cv2.rectangle(images, (0, 0), (213, 70), (255, 255, 255), -1)  # frame to draw, top left, bot right, colour, -1
                    cv2.putText(images,'Rotate your hand until',(5, 15),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                    cv2.putText(images,'your palm faces the',(5, 40),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                    cv2.putText(images,'floor',(5, 65),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                elif(curr_rot==2 and (highest_axis_index!=0 or highest_list_values>0)):
                    # Draw a rectangle
                    images = cv2.rectangle(images, (0, 0), (213, 70), (255, 255, 255), -1)
                    cv2.putText(images,'Rotate your hand until',(5, 15),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                    cv2.putText(images,'your palm faces the',(5, 40),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                    cv2.putText(images,'ceiling',(5, 65),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                # elbow
                if(user_angle234<guidance_end_angle234-end_angle234_allowance):
                    images = cv2.rectangle(images, (225, 340) , (425, 360) , (255, 255, 255), -1)
                    cv2.putText(images,'Straighten your arm',(230, 355),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                # up, down
                if(user_angle814<guidance_end_angle814-end_angle814_lowerAllowance):
                    images = cv2.rectangle(images, (228, 0) , (405, 20) , (255, 255, 255), -1)
                    cv2.putText(images,'Move your arm up',(232, 15),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                elif (user_angle814>guidance_end_angle814+end_angle814_upperAllowance):
                    images = cv2.rectangle(images, (216, 0) , (406, 20) , (255, 255, 255), -1)
                    cv2.putText(images,'Move your arm down',(218, 15),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                # front, back
                if(planar_angle_chest_13<guidance_planar_angle_chest_13-planar_angle_chest_13_lowerAllowance and planar_angle_chest_14<guidance_planar_angle_chest_14-planar_angle_chest_14_lowerAllowance):
                    images = cv2.rectangle(images, (410, 0) ,(640, 20) , (255, 255, 255), -1)
                    cv2.putText(images,'Move your arm backward',(413, 15),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                elif(planar_angle_chest_13>guidance_planar_angle_chest_13+planar_angle_chest_13_upperAllowance and planar_angle_chest_14>guidance_planar_angle_chest_14+planar_angle_chest_14_upperAllowance):
                    images = cv2.rectangle(images, (420, 0) ,(640, 20) , (255, 255, 255), -1)
                    cv2.putText(images,'Move your arm forward',(425, 15),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                else:
                    images = cv2.rectangle(images, (420, 0) ,(640, 20) , (255, 255, 255), -1)
                    cv2.putText(images,'Straighten your elbow',(425, 15),font, 0.5,(0, 0, 255),1,cv2.LINE_4)

            else: # within tolerance, green colour
                #0 - 1
                cv2.line(images, (int(guidance_skele_part_end[0]['0'][0]),int(guidance_skele_part_end[0]['0'][1])), (int(guidance_skele_part_end[0]['1'][0]),int(guidance_skele_part_end[0]['1'][1])), colour_green, thickness=line_thickness)
                #1 - 2
                cv2.line(images, (int(guidance_skele_part_end[0]['2'][0]),int(guidance_skele_part_end[0]['2'][1])), (int(guidance_skele_part_end[0]['1'][0]),int(guidance_skele_part_end[0]['1'][1])), colour_green, thickness=line_thickness)
                #1 - 5
                cv2.line(images, (int(guidance_skele_part_end[0]['5'][0]),int(guidance_skele_part_end[0]['5'][1])), (int(guidance_skele_part_end[0]['1'][0]),int(guidance_skele_part_end[0]['1'][1])), colour_green, thickness=line_thickness)
                #1 - 8
                cv2.line(images, (int(guidance_skele_part_end[0]['8'][0]),int(guidance_skele_part_end[0]['8'][1])), (int(guidance_skele_part_end[0]['1'][0]),int(guidance_skele_part_end[0]['1'][1])), colour_green, thickness=line_thickness)
                #2 - 3
                cv2.line(images, (int(guidance_skele_part_end[0]['2'][0]),int(guidance_skele_part_end[0]['2'][1])), (int(guidance_skele_part_end[0]['3'][0]),int(guidance_skele_part_end[0]['3'][1])), colour_green, thickness=line_thickness)
                #3 - 4
                cv2.line(images, (int(guidance_skele_part_end[0]['3'][0]),int(guidance_skele_part_end[0]['3'][1])), (int(guidance_skele_part_end[0]['4'][0]),int(guidance_skele_part_end[0]['4'][1])), colour_green, thickness=line_thickness)
                #5 - guidance_skele_part_end
                cv2.line(images, (int(guidance_skele_part_end[0]['5'][0]),int(guidance_skele_part_end[0]['5'][1])), (int(guidance_skele_part_end[0]['6'][0]),int(guidance_skele_part_end[0]['6'][1])), colour_green, thickness=line_thickness)
                #6 - 7
                cv2.line(images, (int(guidance_skele_part_end[0]['6'][0]),int(guidance_skele_part_end[0]['6'][1])), (int(guidance_skele_part_end[0]['7'][0]),int(guidance_skele_part_end[0]['7'][1])), colour_green, thickness=line_thickness)
                #8 - 9
                cv2.line(images, (int(guidance_skele_part_end[0]['8'][0]),int(guidance_skele_part_end[0]['8'][1])), (int(guidance_skele_part_end[0]['9'][0]),int(guidance_skele_part_end[0]['9'][1])), colour_green, thickness=line_thickness)
                #8 - 12
                cv2.line(images, (int(guidance_skele_part_end[0]['8'][0]),int(guidance_skele_part_end[0]['8'][1])), (int(guidance_skele_part_end[0]['12'][0]),int(guidance_skele_part_end[0]['12'][1])), colour_green, thickness=line_thickness)
                #9 - 10
                cv2.line(images, (int(guidance_skele_part_end[0]['9'][0]),int(guidance_skele_part_end[0]['9'][1])), (int(guidance_skele_part_end[0]['10'][0]),int(guidance_skele_part_end[0]['10'][1])), colour_green, thickness=line_thickness)
                #10 - 11
                cv2.line(images, (int(guidance_skele_part_end[0]['10'][0]),int(guidance_skele_part_end[0]['10'][1])), (int(guidance_skele_part_end[0]['11'][0]),int(guidance_skele_part_end[0]['11'][1])), colour_green, thickness=line_thickness)
                #12 - 13
                cv2.line(images, (int(guidance_skele_part_end[0]['12'][0]),int(guidance_skele_part_end[0]['12'][1])), (int(guidance_skele_part_end[0]['13'][0]),int(guidance_skele_part_end[0]['13'][1])), colour_green, thickness=line_thickness)
                #13 - 14
                cv2.line(images, (int(guidance_skele_part_end[0]['13'][0]),int(guidance_skele_part_end[0]['13'][1])), (int(guidance_skele_part_end[0]['14'][0]),int(guidance_skele_part_end[0]['14'][1])), colour_green, thickness=line_thickness)
                #CHANGE TO START POSE AFTER PART TWO OF ROTATION
                if(curr_rot == 1):
                    curr_rot=2
                elif(curr_rot == 2):
                    curr_pose=0
                    curr_rot=0


        else: # starting pose
            if(((user_angle123<guidance_start_angle123-start_angle123_lowerAllowance or user_angle123>guidance_start_angle123+start_angle123_upperAllowance) or user_angle234<guidance_start_angle234-start_angle234_allowance) or (highest_axis_index!=1 or highest_list_values<0) or
                bol_wrong ==  True): # outside tolerance, red colour
                #0 - 1
                cv2.line(images, (int(guidance_skele_part_start[0]['0'][0]),int(guidance_skele_part_start[0]['0'][1])), (int(guidance_skele_part_start[0]['1'][0]),int(guidance_skele_part_start[0]['1'][1])), colour_red, thickness=line_thickness)
                #1 - 2
                cv2.line(images, (int(guidance_skele_part_start[0]['2'][0]),int(guidance_skele_part_start[0]['2'][1])), (int(guidance_skele_part_start[0]['1'][0]),int(guidance_skele_part_start[0]['1'][1])), colour_red, thickness=line_thickness)
                #1 - 5
                cv2.line(images, (int(guidance_skele_part_start[0]['5'][0]),int(guidance_skele_part_start[0]['5'][1])), (int(guidance_skele_part_start[0]['1'][0]),int(guidance_skele_part_start[0]['1'][1])), colour_red, thickness=line_thickness)
                #1 - 8
                cv2.line(images, (int(guidance_skele_part_start[0]['8'][0]),int(guidance_skele_part_start[0]['8'][1])), (int(guidance_skele_part_start[0]['1'][0]),int(guidance_skele_part_start[0]['1'][1])), colour_red, thickness=line_thickness)
                #2 - 3
                cv2.line(images, (int(guidance_skele_part_start[0]['2'][0]),int(guidance_skele_part_start[0]['2'][1])), (int(guidance_skele_part_start[0]['3'][0]),int(guidance_skele_part_start[0]['3'][1])), colour_red, thickness=line_thickness)
                #3 - 4
                cv2.line(images, (int(guidance_skele_part_start[0]['3'][0]),int(guidance_skele_part_start[0]['3'][1])), (int(guidance_skele_part_start[0]['4'][0]),int(guidance_skele_part_start[0]['4'][1])), colour_red, thickness=line_thickness)
                #5 - 6
                cv2.line(images, (int(guidance_skele_part_start[0]['5'][0]),int(guidance_skele_part_start[0]['5'][1])), (int(guidance_skele_part_start[0]['6'][0]),int(guidance_skele_part_start[0]['6'][1])), colour_red, thickness=line_thickness)
                #6 - 7
                cv2.line(images, (int(guidance_skele_part_start[0]['6'][0]),int(guidance_skele_part_start[0]['6'][1])), (int(guidance_skele_part_start[0]['7'][0]),int(guidance_skele_part_start[0]['7'][1])), colour_red, thickness=line_thickness)
                #8 - 9
                cv2.line(images, (int(guidance_skele_part_start[0]['8'][0]),int(guidance_skele_part_start[0]['8'][1])), (int(guidance_skele_part_start[0]['9'][0]),int(guidance_skele_part_start[0]['9'][1])), colour_red, thickness=line_thickness)
                #8 - 12
                cv2.line(images, (int(guidance_skele_part_start[0]['8'][0]),int(guidance_skele_part_start[0]['8'][1])), (int(guidance_skele_part_start[0]['12'][0]),int(guidance_skele_part_start[0]['12'][1])), colour_red, thickness=line_thickness)
                #9 - 10
                cv2.line(images, (int(guidance_skele_part_start[0]['9'][0]),int(guidance_skele_part_start[0]['9'][1])), (int(guidance_skele_part_start[0]['10'][0]),int(guidance_skele_part_start[0]['10'][1])), colour_red, thickness=line_thickness)
                #10 - 11
                cv2.line(images, (int(guidance_skele_part_start[0]['10'][0]),int(guidance_skele_part_start[0]['10'][1])), (int(guidance_skele_part_start[0]['11'][0]),int(guidance_skele_part_start[0]['11'][1])), colour_red, thickness=line_thickness)
                #12 - 13
                cv2.line(images, (int(guidance_skele_part_start[0]['12'][0]),int(guidance_skele_part_start[0]['12'][1])), (int(guidance_skele_part_start[0]['13'][0]),int(guidance_skele_part_start[0]['13'][1])), colour_red, thickness=line_thickness)
                #13 - 14
                cv2.line(images, (int(guidance_skele_part_start[0]['13'][0]),int(guidance_skele_part_start[0]['13'][1])), (int(guidance_skele_part_start[0]['14'][0]),int(guidance_skele_part_start[0]['14'][1])), colour_red, thickness=line_thickness)

                #rotation
                if(highest_axis_index!=1 or highest_list_values<0):
                    images = cv2.rectangle(images, (0, 0), (235, 50), (255, 255, 255), -1)  # frame to draw, top left, bot right, colour, -1
                    cv2.putText(images,'Rotate your hand until',(5, 15),font, 0.5,(0, 0, 255),1,cv2.LINE_4)
                    cv2.putText(images,'your palm faces your leg',(5, 40),font, 0.5,(0, 0, 255),1,cv2.LINE_4)



            else: # within tolerance, green colour
                #0 - 1
                cv2.line(images, (int(guidance_skele_part_start[0]['0'][0]),int(guidance_skele_part_start[0]['0'][1])), (int(guidance_skele_part_start[0]['1'][0]),int(guidance_skele_part_start[0]['1'][1])), colour_green, thickness=line_thickness)
                #1 - 2
                cv2.line(images, (int(guidance_skele_part_start[0]['2'][0]),int(guidance_skele_part_start[0]['2'][1])), (int(guidance_skele_part_start[0]['1'][0]),int(guidance_skele_part_start[0]['1'][1])), colour_green, thickness=line_thickness)
                #1 - 5
                cv2.line(images, (int(guidance_skele_part_start[0]['5'][0]),int(guidance_skele_part_start[0]['5'][1])), (int(guidance_skele_part_start[0]['1'][0]),int(guidance_skele_part_start[0]['1'][1])), colour_green, thickness=line_thickness)
                #1 - 8
                cv2.line(images, (int(guidance_skele_part_start[0]['8'][0]),int(guidance_skele_part_start[0]['8'][1])), (int(guidance_skele_part_start[0]['1'][0]),int(guidance_skele_part_start[0]['1'][1])), colour_green, thickness=line_thickness)
                #2 - 3
                cv2.line(images, (int(guidance_skele_part_start[0]['2'][0]),int(guidance_skele_part_start[0]['2'][1])), (int(guidance_skele_part_start[0]['3'][0]),int(guidance_skele_part_start[0]['3'][1])), colour_green, thickness=line_thickness)
                #3 - 4
                cv2.line(images, (int(guidance_skele_part_start[0]['3'][0]),int(guidance_skele_part_start[0]['3'][1])), (int(guidance_skele_part_start[0]['4'][0]),int(guidance_skele_part_start[0]['4'][1])), colour_green, thickness=line_thickness)
                #5 - 6
                cv2.line(images, (int(guidance_skele_part_start[0]['5'][0]),int(guidance_skele_part_start[0]['5'][1])), (int(guidance_skele_part_start[0]['6'][0]),int(guidance_skele_part_start[0]['6'][1])), colour_green, thickness=line_thickness)
                #6 - 7
                cv2.line(images, (int(guidance_skele_part_start[0]['6'][0]),int(guidance_skele_part_start[0]['6'][1])), (int(guidance_skele_part_start[0]['7'][0]),int(guidance_skele_part_start[0]['7'][1])), colour_green, thickness=line_thickness)
                #8 - 9
                cv2.line(images, (int(guidance_skele_part_start[0]['8'][0]),int(guidance_skele_part_start[0]['8'][1])), (int(guidance_skele_part_start[0]['9'][0]),int(guidance_skele_part_start[0]['9'][1])), colour_green, thickness=line_thickness)
                #8 - 12
                cv2.line(images, (int(guidance_skele_part_start[0]['8'][0]),int(guidance_skele_part_start[0]['8'][1])), (int(guidance_skele_part_start[0]['12'][0]),int(guidance_skele_part_start[0]['12'][1])), colour_green, thickness=line_thickness)
                #9 - 10
                cv2.line(images, (int(guidance_skele_part_start[0]['9'][0]),int(guidance_skele_part_start[0]['9'][1])), (int(guidance_skele_part_start[0]['10'][0]),int(guidance_skele_part_start[0]['10'][1])), colour_green, thickness=line_thickness)
                #10 - 11
                cv2.line(images, (int(guidance_skele_part_start[0]['10'][0]),int(guidance_skele_part_start[0]['10'][1])), (int(guidance_skele_part_start[0]['11'][0]),int(guidance_skele_part_start[0]['11'][1])), colour_green, thickness=line_thickness)
                #12 - 13
                cv2.line(images, (int(guidance_skele_part_start[0]['12'][0]),int(guidance_skele_part_start[0]['12'][1])), (int(guidance_skele_part_start[0]['13'][0]),int(guidance_skele_part_start[0]['13'][1])), colour_green, thickness=line_thickness)
                #13 - 14
                cv2.line(images, (int(guidance_skele_part_start[0]['13'][0]),int(guidance_skele_part_start[0]['13'][1])), (int(guidance_skele_part_start[0]['14'][0]),int(guidance_skele_part_start[0]['14'][1])), colour_green, thickness=line_thickness)
                #CHANGE TO END POSE
                curr_pose = 1
                curr_rot = 1
                if(bol_start != True):
                    counter += 1

                bol_start = False

        csv_frames += 1

        # print("user_angle123= ", user_angle123)
        # print("user_angle234= ", user_angle234)
        # print("user_angle813= ", user_angle813)
        # print("planar_angle_chest= ", planar_angle_chest)
        # print("Arduino= ", highest_axis_index, " ", highest_list_values)

        key = cv2.waitKey(1)
        if key == ord('h') and help_switch == 0:
            help_switch = 1
        elif key == ord('h') and help_switch == 1:
            help_switch = 0

        # for the user guide
        cv2.putText(images,'press h for help', (490, 350), font, 0.5,  (0, 0, 255),  1,  cv2.LINE_4)

        # draw circle w/ line across
        images = cv2.circle(images, (600, 75), 36, (0, 0, 0), -1)
        images = cv2.line(images, (630,55), (578,100), (255,255,255), 2)
        cv2.putText(images,  str(rep_value),  (603, 95),  font, 0.6,  (255, 255, 255),  1,  cv2.LINE_4)
        cv2.putText(images,  str(counter),  (588, 68),  font, 0.6,  (255, 255, 255),  1,  cv2.LINE_4)

        # for the help screen
        if help_switch == 1:
            images = cv2.rectangle(images, (0,0), (640,360), (255, 255, 255), -1)
            cv2.putText(images, 'User guide:',(5,15), font, 0.4, (0, 0, 0), 1, cv2.LINE_4)
            images = cv2.line(images, (5,17), (85,17), (0,0,0), 1)
            cv2.putText(images, 'Blue',(5,30), font, 0.4, (255, 0, 0), 1, cv2.LINE_4)
            cv2.putText(images, 'circles represent your',  (40,30), font, 0.4,  (0, 0, 0),  1,  cv2.LINE_4)
            cv2.putText(images,  'skeleton.',  (205,30), font, 0.4,  (255, 0, 0),  1,  cv2.LINE_4)
            cv2.putText(images,  'The',  (5,45),  font, 0.4,  (0, 0, 0),  1,  cv2.LINE_4)
            cv2.putText(images,  'red',  (35,45), font, 0.4,  (0, 0, 255), 1,  cv2.LINE_4)
            cv2.putText(images, 'lines represent the', (65,45), font, 0.4, (0, 0, 0),  1,  cv2.LINE_4)
            cv2.putText(images, 'correct',  (205,45), font, 0.4, (0, 0, 255),  1, cv2.LINE_4)
            cv2.putText(images, 'pose.',  (265,45), font, 0.4, (0, 0, 0), 1, cv2.LINE_4)
            cv2.putText(images,  'If the',  (5,60), font, 0.4,  (0, 0, 0), 1,  cv2.LINE_4)
            cv2.putText(images,  'red',  (50,60), font, 0.4,  (0, 0, 255),  1,  cv2.LINE_4)
            cv2.putText(images, 'lines turn',  (80,60), font, 0.4,  (0, 0, 0),  1,  cv2.LINE_4)
            cv2.putText(images, 'green',  (160,60), font, 0.4, (34, 139, 34), 1,  cv2.LINE_4)
            cv2.putText(images,', correct pose is', (200,60), font, 0.4, (0, 0, 0), 1,  cv2.LINE_4)
            cv2.putText(images,  'achieved!',  (320,60), font, 0.4,  (0, 69, 255), 1,  cv2.LINE_4)
            cv2.putText(images,  'If not, warning messages will appear to tell you how to correct it:',  (5, 75),  font, 0.4, (0, 0, 0),  1, cv2.LINE_4)
            cv2.putText(images,  '- Top left: rotation of arms',  (5, 95),  font, 0.4,  (0, 0, 0),  1,  cv2.LINE_4)
            cv2.putText(images,  '- Top middle: up or down movement of the arms',  (5, 110),  font, 0.4,  (0, 0, 0),  1,  cv2.LINE_4)
            cv2.putText(images,  '- Top right: front or back movement of the arms',  (5, 125), font, 0.4,  (0, 0, 0), 1,  cv2.LINE_4)
            cv2.putText(images, '- Bottom left: how far you are from the camera', (5, 140), font, 0.4, (0, 0, 0), 1, cv2.LINE_4)
            cv2.putText(images,  '- Bottom middle: elbow corrections',  (5, 155),  font, 0.4,  (0, 0, 0), 1, cv2.LINE_4)
            cv2.putText(images, 'The number at the top right indicates how many times the exercise is completed.', (5, 175),  font, 0.4,  (0, 0, 0),  1, cv2.LINE_4)
            cv2.putText(images,  'How to:',  (5,200), font, 0.4,  (0, 0, 0),  1,  cv2.LINE_4)
            images = cv2.line(images, (5,202), (55,202), (0,0,0), 1)
            cv2.putText(images,  '1) Go through the gif in the selection screen before proceeding with the exercise.',  (5, 215),  font, 0.4,  (0, 0, 0),  1, cv2.LINE_4)
            cv2.putText(images,  '2) Stand 2 to 3m away from the camera.',  (5, 230),  font, 0.4,  (0, 0, 0),  1, cv2.LINE_4)
            cv2.putText(images,  '3) The correct pose is shown by the red skeleton in the middle of the screen.',  (5, 245),  font, 0.4,  (0, 0, 0),  1, cv2.LINE_4)
            cv2.putText(images,  '4) There is no need to align your skeleton to the red skeleton.',  (5, 260),  font, 0.4,  (0, 0, 0),  1, cv2.LINE_4)
            cv2.putText(images,  "5) Follow the red skeleton's pose.",  (5, 275),  font, 0.4,  (0, 0, 0),  1, cv2.LINE_4)
            cv2.putText(images,  '6) Any difference between your skeleton and the red skeleton will be shown as',  (5, 290),  font, 0.4,  (0, 0, 0),  1, cv2.LINE_4)
            cv2.putText(images,  'warnings around the screen.',  (5, 305),  font, 0.4,  (0, 0, 0),  1, cv2.LINE_4)
            cv2.putText(images,  '7) When one cycle is completed, the counter on the right will increment by one.',  (5, 320),  font, 0.4,  (0, 0, 0),  1, cv2.LINE_4)
            cv2.putText(images,  'press h to exit',  (525, 355),  font, 0.4,  (0, 0, 255),  1,  cv2.LINE_4)

        # make screen full screen
        cv2.namedWindow("RealSense", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("RealSense",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

        # Show images
        cv2.imshow("RealSense", images)
        # key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            # sys.exit()
            # cv2.destroyAllWindows()
            cv2.destroyWindow("RealSense")
            break
        # elif key == ord('h') and help_switch == 0:
        #     help_switch = 1
        # elif key == ord('h') and help_switch == 1:
        #     help_switch = 0


finally:

    # Stop streaming
    pipeline.stop()
