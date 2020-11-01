import cv2
import dlib
import time
import numpy as np
import sys
import pickle
import virtualvideo

# scrub through a video
# collect all the faces
# cluster them by person
# within person, cluster them by pose.
# save a representative sample of each (person, pose) to folder.

class Detector:
    def __init__(self):
        self.scale = 1
        # the CNN works better than the frontal face detector
        self.detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        #self.detector = dlib.get_frontal_face_detector()
        self.facemarker = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def faces(self, img):
        img = cv2.resize(img, (img.shape[1]//self.scale, img.shape[0]//self.scale))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.detector(gray)
        results = []
        for det in detections:
            shape = self.facemarker(img, det.rect)
            """
            pose = np.array([
                x
                for part in shape.parts()
                for x in [part.x-shape.rect.left(), part.y-shape.rect.top()]
            ])"""

            x_range = [pt.x for pt in shape.parts()]
            xmin = min(x_range)
            xwidth = max(x_range)-min(x_range)
            y_range = [pt.y for pt in shape.parts()]
            ymin = min(y_range)
            ywidth = max(y_range)-min(y_range)

            pose = np.array([
                x
                for part in shape.parts()
                for x in [(part.x-xmin)/xwidth,(part.y-ymin)/ywidth]
            ])


            bbox = np.array([
                shape.rect.left(),
                shape.rect.top(),
                shape.rect.width(),
                shape.rect.height()
            ])
            results.append( (pose, bbox) )
        return results

def pasteimage(img, dest_bbox, new_img):
    dest_size = (int(round(dest_bbox[3] * new_img.shape[0] / new_img.shape[1])), dest_bbox[3])
    #print(dest_bbox, new_img.shape, dest_size, img.shape)
    new_img = cv2.resize(new_img, dest_size)
    #print("result: ", new_img.shape)

    w = min(dest_bbox[2], new_img.shape[1])
    h = min(dest_bbox[3], new_img.shape[0])
    #print("wh: ",w,h)

    dst_x = max(0, dest_bbox[0])
    dst_y = max(0, dest_bbox[1])
    dst_r = min(img.shape[1], dest_bbox[0]+w)
    dst_b = min(img.shape[0], dest_bbox[1]+h)
    h = dst_b - dst_y
    w = dst_r - dst_x
    #print(dst_x, dst_y, dst_r, dst_b, h, w)

    src_x = dst_x - dest_bbox[0]
    src_y = dst_y - dest_bbox[1]

    img[dst_y:dst_b, dst_x:dst_r] = new_img[src_y:src_y+h, src_x:src_x+w]

class Source(virtualvideo.VideoSource):
    def img_size(self):
        return (640, 480)
    def fps(self):
        return 30
    def pix_fmt(self):
        return "bgr24"

    def generator(self):
        input_file = sys.argv[1]
        cam = cv2.VideoCapture("/dev/video2")
        vid = cv2.VideoCapture(input_file)
        with open(input_file+".poses", "rb") as f:
            poses_by_label = pickle.load(f)
        current_label = list(poses_by_label.keys())[0]
        full = np.array([p[0] for p in poses_by_label[current_label]])

        x_range = full[:, ::2]
        xmin = x_range.min(axis=1)
        xwidth = x_range.max(axis=1)-xmin
        full[:, ::2] = (x_range - xmin.reshape(x_range.shape[0], 1))/xwidth.reshape(x_range.shape[0], 1)
        y_range = full[:, 1::2]
        ymin = y_range.min(axis=1)
        ywidth = y_range.max(axis=1)-ymin
        full[:, 1::2] = (y_range - ymin.reshape(y_range.shape[0], 1))/ywidth.reshape(y_range.shape[0], 1)



        print( [(k, len(v)) for k,v in poses_by_label.items()] )
        print("on label {} count = {}".format(current_label, len(poses_by_label[current_label])))
        d = Detector()
        t2 = time.time()
        closest_i = 0
        dvd = False
        dvdlogo = cv2.imread("dvdlogo.png")


        cv2.startWindowThread()
        while True:
            ok, img = cam.read()
            if not ok:
                print("framedrop")
                continue
            faces = d.faces(img)
            for pose, bbox in faces:
                if dvd:
                    pasteimage(img, bbox, dvdlogo)
                if not dvd:
                    if time.time() - t2 > 1:
                        t2 = time.time()
                        closest_i = np.linalg.norm(full-pose, axis=1).argmin()
                        #print("newface", closest_i, poses_by_label[current_label][closest_i])

                    _, face_bbox, frame_i = poses_by_label[current_label][closest_i]
                    vid.set(cv2.CAP_PROP_POS_FRAMES, max(frame_i-50,0))
                    while vid.get(cv2.CAP_PROP_POS_FRAMES) < frame_i:
                        vid.read()
                    ok2, vidframe = vid.read()
                    if not ok2:
                        print("vid drop")
                        continue
                    x,y,w,h=face_bbox
                    faceimg = vidframe[y:y+h,x:x+w]
                    vidframe= cv2.rectangle(vidframe, face_bbox, (128,128,128))
                    cv2.imshow("vid", vidframe)
                    if w > 0 and h > 0 and faceimg.shape[0] > 0 and faceimg.shape[1] > 0:
                        cv2.imshow("face", faceimg)
                        pasteimage(img, bbox, faceimg)
            cv2.imshow('test', img)
            r=cv2.waitKey(1)
            if r == ord('n'):
                current_label = (current_label + 1) % len(poses_by_label)
                t2 = time.time()-10
                full = np.array([p[0] for p in poses_by_label[current_label]])

                x_range = full[:, ::2]
                xmin = x_range.min(axis=1)
                xwidth = x_range.max(axis=1)-xmin
                full[:, ::2] = (x_range - xmin.reshape(x_range.shape[0], 1))/xwidth.reshape(x_range.shape[0], 1)
                y_range = full[:, 1::2]
                ymin = y_range.min(axis=1)
                ywidth = y_range.max(axis=1)-ymin
                full[:, 1::2] = (y_range - ymin.reshape(y_range.shape[0], 1))/ywidth.reshape(y_range.shape[0], 1)


                print("on label {} count = {}".format(current_label, len(poses_by_label[current_label])))
            if r == ord('d'):
                dvd = not dvd
            yield(img)

src=Source()
out=virtualvideo.FakeVideoDevice()
out.init_input(src)
out.init_output(9, 640, 480, fps=30, pix_fmt="yuyv422")

print("start")
try:
    out.run()
except KeyboardInterrupt:
    out.stop()
