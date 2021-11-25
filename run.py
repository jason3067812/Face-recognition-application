
import cv2
import numpy as np
from pulse import Pulse
import time
from threading import Lock, Thread
from plot_cont import DynamicPlot
from capture_frames import CaptureFrames
from process_mask import ProcessMasks

from utils import *
import multiprocessing as mp
import sys
from optparse import OptionParser

from matplotlib import pyplot as plt
from models.detector import face_detector
from models.detector.iris_detector import IrisDetector
import warnings
warnings.filterwarnings("ignore")
import os

class RunPOS():
    def __init__(self,  sz=270, fs=28, bs=30, plot=False, rs=(256,256), it=0.7):
        self.batch_size = bs
        self.frame_rate = fs
        self.signal_size = sz
        self.plot = plot
        self.resize_size = rs
        self.iou_threshold = it

    def __call__(self, source):
        time1=time.time()
        
        mask_process_pipe, chil_process_pipe = mp.Pipe()
        self.plot_pipe = None
        if self.plot:
            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plotter = DynamicPlot(self.signal_size, self.batch_size)
            self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
            self.plot_process.start()
        
        process_mask = ProcessMasks(self.signal_size, self.frame_rate, self.batch_size)

        mask_processer = mp.Process(target=process_mask, args=(chil_process_pipe, self.plot_pipe, source, ), daemon=True)
        mask_processer.start()
        
        capture = CaptureFrames(self.batch_size, source, self.resize_size , self.iou_threshold, show_mask=True)
        capture(mask_process_pipe, source)

        mask_processer.join()
        if self.plot:
            self.plot_process.join()

        time2=time.time()
        print(f'time {time2-time1}')


def get_args():

    parser = OptionParser()
    parser.add_option('-s', '--source', dest='source', default=0,type='int',
                        help='Signal Source: 0 for webcam or file path')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=30,
                        type='int', help='batch size')
    parser.add_option('-f', '--frame-rate', dest='framerate', default=25,
                        help='Frame Rate')
    parser.add_option('-r', '--resize-size', dest='resizesize', default=(128,128),
                      help='Resize Size')
    parser.add_option('-i', '--iou-threshold', dest='iouthreshold', default=0.7,
                      help='Iou Threshold')
    (options, _) = parser.parse_args()
    return options


if __name__=="__main__":


    args = get_args()
    source = args.source
    #source = "C:\\Users\\jason\\face_toolbox_keras-master\\video\\a.mp4"

    runPOS = RunPOS(270, args.framerate, args.batchsize, True, args.resizesize, args.iouthreshold)
    runPOS(source)
    