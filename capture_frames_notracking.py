import cv2
import numpy as np
import torch
from torch import nn
from model import LinkNet34
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageFilter
import time
import sys

from matplotlib import pyplot as plt
from models.detector import face_detector
from models.verifier.face_verifier import FaceVerifier
from models.verifier.face_verifier_tflite import FaceVerifier_tflite
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.keras.models import load_model



class CaptureFrames():


    def __init__(self, bs, source, show_mask=False):
        self.frame_counter = 0
        self.batch_size = bs
        self.stop = False
        self.device = torch.device("cpu")    #"cuda:0" if torch.cuda.is_available() else
        # self.model = LinkNet34()
        # self.model.load_state_dict(torch.load('linknet.pth'))

        # add1 : linknet pytorch to keras
        self.model = load_model(r'C:\Users\jason\face_toolbox_keras-master\linknet_keras.h5')   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # self.model.eval()

        # self.model.to(self.device)
        self.show_mask = show_mask

        self.fd = face_detector.FaceAlignmentDetector(fd_type="s3fd")

        print("building face database....")

        self.fv = FaceVerifier(classes=512, extractor="insightface")  # extractor="insightface", "facenet"
        self.fv.set_detector(self.fd)
        self.a, self.b = self.fv.build_face_identity_database(r"C:\Users\jason\face_toolbox_keras-master\images\0",
                                                              with_detection=True, with_alignment=False)

        # tflite
        # self.fvlite = FaceVerifier_tflite(classes=512, extractor="insightface-tflite")
        # self.fvlite.set_detector(self.fd)
        # self.a, self.b = self.fvlite.build_face_identity_database(r"C:\Users\jason\face_toolbox_keras-master\images\0")


        print(self.a)


    def __call__(self, pipe, source):

        self.pipe = pipe
        self.capture_frames(source)


    def capture_frames(self, source):


        name = input("enter name:")


        print(" ")
        print("start detect!")
        print(" ")

        img_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        camera = cv2.VideoCapture(source)
        time.sleep(1)
        #self.model.eval()
        (grabbed, frame) = camera.read()

        time_1 = time.time()
        self.frames_count = 0
        count=0
        while grabbed:

            time_1 = time.time()

            count = count+1


            (grabbed, orig) = camera.read()
            if not grabbed:
                continue
            
            shape = orig.shape[0:2]

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.terminate(camera)
                break
            if cv2.waitKey(1) & 0xFF == ord('a'):
                name = input("enter new name:")


            # add2 : detect face region

            #im = self.resize_image(orig)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            orig_c = orig.copy()
            im = cv2.resize(orig_c,(256,256))
            resize_shape = im.shape


            startaaa = time.time()
            bboxes = self.fd.detect_face(im, with_landmarks=False)
            endaaa = time.time()
            # print("s3fd cost time: ", endaaa - startaaa)

            number = len(bboxes)
            print(f"detect {number} bbox(es)")

            # 開始走訪所有偵測到的臉
            for i in range(number):
                # 處理臉部 bbox

                if bboxes[i] != []:

                    x0, y0, x1, y1, score = bboxes[i]
                    x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])

                    crop_face = im[x0:x1, y0:y1]
                    size = crop_face.shape

                    if size[0] != 0 and size[1] != 0:

                        dist, min_dist, result = self.fv.webcam_verify(crop_face, self.a, self.b,
                                                                       with_alignment=False,
                                                                       threshold=0.6)

                        #dist, min_dist, result = self.fvlite.webcam_verify(crop_face, self.a, self.b, threshold=0)

                        if result == name:

                            print(result)
                            print(" ")

                            # add3 : make mask image
                            mask_background = np.zeros([resize_shape[0], resize_shape[1]], dtype=np.uint8)

                            # s3fd
                            #mask_background[x0 + 20:x1 + 10, y0 - 20:y1 + 20] = 255

                            # mtcnn
                            mask_background[x0 :x1 , y0 :y1 ] = 255
                            

                            mask_frame = cv2.add(im, np.zeros(np.shape(im), dtype=np.uint8), mask=mask_background)
                            # mask_frame = cv2.resize(mask_frame, (256, 256), cv2.INTER_LINEAR)
                            #
                            cv2.imwrite(f"D:/test3/{count}.jpg",mask_frame)
                            #
                            #
                            # sss = time.time()
                            # a = img_transform(Image.fromarray(mask_frame))
                            # a = a.unsqueeze(0)
                            # imgs = Variable(a.to(dtype=torch.float, device=self.device))
                            # # add3 : numpy to torch tensor
                            # pred = self.model.predict(imgs)
                            # pred = torch.from_numpy(pred)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            # # pred = self.model(imgs)
                            # pred = torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
                            # mask = pred.data.cpu().numpy()
                            # mask = mask.squeeze()
                            # mask = mask > 0.8
                            # orig[mask == 0] = 0
                            # ggg = time.time()
                            # print("mask time:",ggg-sss)

                            self.pipe.send([mask_frame])
                            cv2.imshow('demo', mask_frame)  #orig

                        else:
                            print(f"cannot find {name}\n")
                            cv2.imshow('demo', im)
                            continue

                    else:
                        print("wrong crop face\n")
                        cv2.imshow('demo', im)
                        continue

                else:
                    print("cannot detect bbox(es)\n")
                    cv2.imshow('demo', im)
                    continue

            time_2 = time.time()
            print(f"one frame cost:{time_2-time_1}s")


            # if self.show_mask:
            #     cv2.imshow('mask', orig)

            # if self.frames_count % 30 == 29:
            #     time_2 = time.time()
            #     sys.stdout.write('\n\n\n')
            #     sys.stdout.write(f'\rFPS: {30 / (time_2 - time_1)}')
            #     sys.stdout.write('\n\n\n\n\n')
            #     sys.stdout.flush()
            #     time_1 = time.time()

            self.frames_count += 1

            if cv2.waitKey(1) & 0xFF == ord('a'):
                name = input("enter new name:")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.terminate(camera)

        self.terminate(camera)



    def terminate(self, camera):

        self.pipe.send(None)
        cv2.destroyAllWindows()
        camera.release()

    def resize_image(self,im, max_size=768):

        if np.max(im.shape) > max_size:
            ratio = max_size / np.max(im.shape)
            #print(f"Resize image to ({str(int(im.shape[1] * ratio))}, {str(int(im.shape[0] * ratio))}).")
            return cv2.resize(im, (0, 0), fx=ratio, fy=ratio)

        return im
