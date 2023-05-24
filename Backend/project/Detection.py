# from ..core.config import DETECTION_MODEL


# Method-2, load model directly
# detector = insightface.model_zoo.get_model('buffalo_s')
# detector.prepare(ctx_id=0, input_size=(640, 640))
import cv2
import numpy as np
import insightface
import time
from insightface.utils import face_align

from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis("buffalo_s",providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img=cv2.imread("../test/images/Hinh-anh-gai-xinh-deo-kinh-dang-nghieng-dau.jpg")
# cv2.imshow("a",img)
# cv2.waitKey()


faces = app.get(img)
st=time.time()
for face in faces:
    box=face["bbox"].astype("int")
    color = (0, 0, 255)
    point_align=face['kps'].astype("int")
    for p in point_align:

        cv2.circle(img,list(p),3,(0,0,255),1)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

print(faces[0]['kps'])

face_a=face_align.norm_crop(img,faces[0]['kps'],112)

M = face_align.estimate_norm(faces[0]['kps'].astype("int"),112)

new =face_align.trans_points2d(faces[0]['kps'],M)

print(time.time()-st)
box=faces[0]["bbox"].astype("int")

face_img=img[box[1]:box[3],box[0]:box[2]]
for p in new.astype("int"):
    cv2.circle(face_img, list(p), 3, (0, 255, 255), 1)
cv2.imshow("a",face_img)
cv2.waitKey()
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# bboxes, kpss =det_model.detect(img,
#                                      max_num=max_num,
#                                      metric='default')
# if bboxes.shape[0] == 0:
#     return []
# ret = []
# for i in range(bboxes.shape[0]):
#     bbox = bboxes[i, 0:4]