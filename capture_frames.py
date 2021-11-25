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
import tensorflow as tf


class CaptureFrames():

    def __init__(self, bs, source, rs, it, show_mask=False):

        self.frames_count = 0
        self.batch_size = bs
        self.stop = False


        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        tf.keras.backend.set_session(sess)

        self.device = torch.device("cuda:0")
        self.model = LinkNet34()
        self.model.load_state_dict(torch.load('linknet.pth'))
        self.model.eval()
        self.model.to(self.device)
        self.show_mask = show_mask


        self.fd = face_detector.FaceAlignmentDetector(fd_type="s3fd")
        print("building face database....")


        self.fv = FaceVerifier(classes=512, extractor="insightface")  # extractor="insightface", "facenet"
        self.fv.set_detector(self.fd)
        self.a, self.b = self.fv.build_face_identity_database(r"database\front0", with_detection=True,with_alignment=False)
        print("database name list: ",self.a)

        self.img_size = rs
        self.iou_threshold = it
        self.person_identification = "False"
        self.previous_person_identification_location = []

        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.coordinate = "abnormal"


    def __call__(self, pipe, source):

        self.pipe = pipe
        self.capture_frames(source)


    def capture_frames(self, source):

        wanted_name = input("enter name:")

        print(" ")
        print("start detect!")
        print(" ")

        camera = cv2.VideoCapture(source)
        time.sleep(1)
        self.model.eval()
        (grabbed, frame) = camera.read()

        # system start
        while grabbed:

            if cv2.waitKey(1) & 0xFF == ord('s'):
                print(" ")
                wanted_name = input("enter new name:")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.terminate(camera)
                break

            total1 = time.time()

            (grabbed, orig) = camera.read()
            if not grabbed:
                continue

            shape = orig.shape[0:2]
            print("original frame size:",shape)

            im = cv2.resize(orig, self.img_size)

            bboxes, bboxes_number = self.face_detect(im)

            if self.person_identification == "False":

                status = "recognizing"
                print("recognizing")

                bbox = self.face_recognize(bboxes, bboxes_number, im, wanted_name)

                if bbox == []:
                    total2 = time.time()
                    total_time = total2 - total1
                    fps = int(round(1 / total_time, 0))
                    self.draw_information(orig, shape, bbox, status, self.coordinate, wanted_name, fps)
                    cv2.imshow('demo', orig)
                    continue
                else:
                    x0, y0, x1, y1 = bbox

            else:

                status = "tracking"
                print("tracking")

                bbox = self.tracking_algorithm(bboxes, bboxes_number)

                if self.person_identification == "False":
                    print("lose tracking target")
                    total2 = time.time()
                    total_time = total2 - total1
                    fps = int(round(1 / total_time, 0))
                    self.draw_information(orig, shape, bbox, status, self.coordinate, wanted_name, fps)
                    cv2.imshow('demo', orig)
                    continue
                else:
                    x0, y0, x1, y1 = bbox


            mask_frame = self.mask_background(x0, x1, y0, y1, im, interval=0)

            final = self.face_mask(orig, shape, mask_frame, self.img_transform)

            self.pipe.send([final])

            total2 = time.time()
            total_time = total2 - total1
            fps = int(round(1 / total_time, 0))

            self.draw_information(final, shape, bbox, status, self.coordinate, wanted_name, fps)
            cv2.imshow('demo', final)

            print("One frame cost time:", total_time)
            print("FPS:", fps)
            print("\n\n\n")

        self.terminate(camera)






    def terminate(self, camera):

        self.pipe.send(None)
        cv2.destroyAllWindows()
        camera.release()


    def compute_iou(self, rec1, rec2):
        """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect)) * 1.0


    def tracking_algorithm(self, face_bboxes, bboxes_number):

        print("start tracking algorithm")

        iou_list = []
        bbox_list = []
        for j in range(bboxes_number):

            # 處理臉部 bbox
            if face_bboxes[j] != []:
                x0, y0, x1, y1, score = face_bboxes[j]
                x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
                # 防呆: 確保是臉
                if x0 > 0 and y0 > 0 and x1 < self.img_size[0] and y1 < self.img_size[1]:
                    self.coordinate = "normal"

                    iou = self.compute_iou([x0, y0, x1, y1], self.previous_person_identification_location)
                    iou_list.append(iou)
                    bbox = [x0, y0, x1, y1]
                    bbox_list.append(bbox)
                else:
                    self.coordinate = "abnormal"
                    continue

        if iou_list == []:
            self.person_identification = "False"
            return []

        else:

            print("this frame every iou:", iou_list)
            max_iou = max(iou_list)
            print("max iou:", max_iou)

            if max_iou > self.iou_threshold:
                print("iou normal, keep tracking")
                index = iou_list.index(max_iou)
                self.previous_person_identification_location = bbox_list[index]
                return self.previous_person_identification_location

            else:
                print("iou too small, go through face recognition")
                self.person_identification = "False"
                return []



    def face_detect(self, image):

        bboxes = self.fd.detect_face(image, with_landmarks=False)
        bboxes_number = len(bboxes)
        print(f"==> detect {bboxes_number} bbox(es)")
        print(bboxes)

        return bboxes, bboxes_number

    def face_recognize(self, face_bboxes, bboxes_number, image, wanted_name):

        # 走訪所有bboxes
        for i in range(bboxes_number):

            # 處理臉部 bbox
            if face_bboxes[i] != []:

                x0, y0, x1, y1, score = face_bboxes[i]
                x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])

                # 防呆: 確保是正常的框
                if x0 > 0 and y0 > 0 and x1 < self.img_size[0] and y1 < self.img_size[1]:

                    self.coordinate = "normal"

                    crop_face = image[x0:x1, y0:y1]

                    dist, min_dist, result = self.fv.webcam_verify(crop_face, self.a, self.b, with_alignment=False,
                                                                   threshold=0.6)

                    print("辨識到的身分:",result)

                    if result == wanted_name:

                        self.person_identification = "True"
                        self.previous_person_identification_location = [x0, y0, x1, y1]
                        break

                    else:
                        # 繼續走訪下一張臉

                        continue
                else:
                    self.coordinate = "abnormal"
                    continue

            else:
                self.coordinate = "normal"

        if self.person_identification == "False":
            return []
        else:
            return self.previous_person_identification_location


    def mask_background(self, x0, x1, y0, y1, image, interval):

        mask_background = np.zeros(self.img_size, dtype=np.uint8)
        mask_background[x0 - interval: x1 + interval, y0 - interval: y1 + interval] = 255
        mask_frame = cv2.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask=mask_background)

        return mask_frame


    def face_mask(self, original_frame, original_shape, mask_frame, img_transform):

        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2RGB)
        a = img_transform(Image.fromarray(mask_frame))
        a = a.unsqueeze(0)
        imgs = Variable(a.to(dtype=torch.float, device=self.device))
        pred = self.model(imgs)
        pred = torch.nn.functional.interpolate(pred, size=[original_shape[0], original_shape[1]])
        mask = pred.data.cpu().numpy()
        mask = mask.squeeze()
        mask = mask > 0.8
        original_frame[mask == 0] = 0

        return original_frame

    def draw_information(self, image, shape, bbox_status, status, face_coordinate_status, name, fps):


        if bbox_status == []:

            cv2.putText(image, f"cannot find {name}", (10, 40), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 255, 255), 1, cv2.LINE_AA)
            if face_coordinate_status == "abnormal":
                cv2.putText(image, "face has been cut", (10, 80), cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 255, 255), 1, cv2.LINE_AA)
        else:

            if status == "recognizing":
                cv2.putText(image, f"recognizing {name}", (10, 40), cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, f"tracking {name}", (10, 40), cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 255, 255), 1, cv2.LINE_AA)

            test = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            total_pixel = shape[0] * shape[1]
            c = cv2.countNonZero(test)

            if (c / total_pixel) * 100 < 5:
                cv2.putText(image, "please get closer and don't cover your face", (10, 80), cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, f"fps: {fps}", (510, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)







