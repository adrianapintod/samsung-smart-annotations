Python Version - 3.9
cv2 version - 4.5.3

pip3 install opencv-python to install opencv

Steps:
1. Copy Yolo weights from https://pjreddie.com/media/files/yolov3.weights
to Yolo/ Directory

2. From terminal call:
    python yolo_click_crop2.py --image 'Image' --yolo yolo-coco

    Example:
    python yolo_click_crop2.py --image image_0.jpg --yolo yolo-coco
    python yolo_click_crop2.py --image dog.jpg --yolo yolo-coco

3. Double click to crop and crop image is saved in the same directory.
