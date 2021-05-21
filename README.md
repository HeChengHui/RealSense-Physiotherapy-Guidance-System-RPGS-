# RealSense Physiotherapy Guidance System (RPGS)
<p align="center">
  <img width="150" height="150" src="https://user-images.githubusercontent.com/84503515/119081935-9f133f00-ba2f-11eb-8483-a9cad97136bf.png">
</p>

## Introduction
RPGS is a home-based physiotherapy aid that provides real time sensing and guidance for physiotherapy patients to complete their exercises at home. It consists of three synergistic components: Intel RealSense D435 3D camera, a wearable module consisting of an Arduino UNO and a MPU-6050 Six-Axis (Gyro + Accelerometer) MEMS MotionTracking™ Device, and an interactive GUI that provides feedback to the patient in real time.

Currently, this system only work for 1 exercise (right arm raise and rotation)
![image](https://user-images.githubusercontent.com/84503515/119089767-4185ef00-ba3d-11eb-9747-ad3def3d8385.png)
*NOTE: Between step 3 & 4, palm should face down first before going to the ending pose*

More information can be seen in the group report.

[Project Done in 2020]

## Pre-requisites
*NOTE: wasn't able to get the requirements.txt as of writing
1) Intel RealSense D435 camera along with their developer SDK
2) [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
3) Arduino Uno + MPU-6050 + Arduino IDE + long cable*
4) I believe was done using python 3.6.0
5) PyQt5

*- Long cable needed in this project as we are unable to get the bluetooth module to work. Hence a long wire connecting the arduino to the computer is needed.

## Directory
*NOTE: for some reason I wasn't able to get OpenPose working outside of their tutorial directory. Hence all files are stored inside that directory
```bash
openpose-master\build\examples\tutorial_api_python
├── Dev_Json Folder
├── Pose Database Folder
├── Realsense_colour_images Folder
└── Py + UI files
```
1) Dev_Json  
To store the changed angle and repetition values. 
3) Pose Database  
To store the json files of keypoints of the start and end poses. 
3) Realsense_colour_images  
To store the saved image from realsense to be processed, before getting processed by openpose (delete the text file inside first)

## Others
Added report and reading materials.
Look into aspect ratio if the the skeleton doesn't line up with the person.

## Credits
Special Thanks to WANDERSON M.PIMENTA for his user inteface.  
https://www.youtube.com/channel/UCy1fv5dh3wQEem1nFAUBJzw
