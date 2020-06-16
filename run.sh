#!/bin/sh
python3 modified_draw.py
python3 people_counter_line_last.py --prototxt ./mobilenet_ssd/MobileNetSSD_deploy.prototxt --model ./mobilenet_ssd/MobileNetSSD_deploy.caffemodel --output ./output/output1.mp4 
