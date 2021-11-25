# Side_Project-Face_recognition_application
## This project is the application of face recognition. There are two applications:
1. Multi-angle face recognition
2. RPPG based on face recognition
## Requirements
- Python 3
- Keras 2.2.4
- TensorFlow 1.12.0 or 1.13.1
- Numpy
- Pytorch
- OpenCv
- Matplotlib, Scipy, Pillow
- Please download models file from https://drive.google.com/file/d/1TSWXg9v4-IzWG0gXAKe89Bfqi31B9Y8D/view?usp=sharing. Extract and place it on the outermost folder
## Usage
### - Multi-angle face recognition
1. Enter the database directory and put three face photos with different angles into three corresponding folders
2. Run Face_recognition_v7_Multiple angle detection.ipynb
### - RPPG based on face recognition
1. Run run.py
2. Enter the name you want to detect. This name must be stored in the database
3. enter "q" to exist system ; "s" to change the person you want to detect
4. get result.png in the outermost folder.
## System activity diagram
### - Multi-angle face recognition
<p align="center">
  <img src="https://user-images.githubusercontent.com/56544982/143415234-3af31f5f-1bde-4ca4-83d3-a36ea0aa7e86.png">
</p>

### - RPPG based on face recognition

<p align="center">
  <img  src="https://user-images.githubusercontent.com/56544982/143415359-75c7a9f4-4c8a-4371-8fcb-63622771fc7f.png">
</p>

## Key algorithm
1. Using human faces 68 landmarks to implement face angle detection
2. Increasing frames rate by calculating bbox iou to implement face tracking 

## Interface demo
### - Multi-angle face recognition
#### * Display result
<p align="center">
  <img  src="https://user-images.githubusercontent.com/56544982/143415495-6fc0d3a1-45e7-49a2-ab29-5aee959754b8.png">
</p>

#### * Alert
<p align="center">
  <img  src="https://user-images.githubusercontent.com/56544982/143415757-fa944baa-7880-4dee-ae1f-964dfc90c52b.png">
</p>
</b>

### - RPPG based on face recognition
#### * Status: tracking
<p align="center">
  <img  src="https://user-images.githubusercontent.com/56544982/143416060-0e7bf2c1-4162-43da-8034-eb2d0b186b74.png">
</p>

##### * Alert
<p align="center">
  <img  src="https://user-images.githubusercontent.com/56544982/143416080-b7476711-d0d9-4641-baa7-f4abaf53f39b.png">
</p>

##### * Display result
<p align="center">
  <img  src="https://user-images.githubusercontent.com/56544982/143416106-ffb51685-2b8d-438b-81ce-1084b7eb2623.png">
</p>


## Reference
I learn a lot from https://github.com/shaoanlu/face_toolbox_keras, and https://github.com/nasir6/rPPG. Really thanks to them~
