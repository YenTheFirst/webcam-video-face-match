import cv2
import dlib
import time
import numpy as np
import sys
import pickle

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
        #self.facemarker = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        self.win = dlib.image_window()
        self.all_faces = []
        self.all_poses = []
        self.frame_mark = []

    def update(self, imgs, base_frame_i):
        #if frame_i % 17 != 0:
        #    return
        print("detect in frame", base_frame_i)
        #img = cv2.resize(img, (img.shape[1]//self.scale, img.shape[0]//self.scale))
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        all_detections = self.detector(imgs)
        for j, detections in enumerate(all_detections):
            frame_i = j+base_frame_i
            if frame_i % 10 == 0:
                self.win.set_image(rgb)
                self.win.clear_overlay()
            for det in detections:
                if det.confidence < 1.0:
                    continue
                shape = self.facemarker(img, det.rect)
                descriptor = self.facerec.compute_face_descriptor(rgb, shape)
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
                self.all_faces.append(descriptor)
                self.all_poses.append((pose, bbox, frame_i))
                self.frame_mark.append(frame_i)
                if frame_i % 10 == 0:
                    self.win.add_overlay(det.rect)
                    self.win.add_overlay(shape)
                #if frame_i >= 340:
                #    cv2.imshow('test', img)
                #    cv2.waitKey(0)

    def cluster(self):
        face_labels = dlib.chinese_whispers_clustering(self.all_faces, 0.5)
        self.pose_by_label = {}
        for i, label in enumerate(face_labels):
            if label not in self.pose_by_label:
                self.pose_by_label[label] = []
            self.pose_by_label[label].append(self.all_poses[i])

if __name__ == "__main__":
    input_file = sys.argv[1]
    cam = cv2.VideoCapture(input_file)
    d = Detector()
    t1 = time.time()
    count = 0
    try:
        while True:
            frame_i = cam.get(cv2.CAP_PROP_POS_FRAMES)
            bufsize=1
            l = [0]*bufsize
            for i in range(bufsize):
                ok, img = cam.read()
                if not ok or img.shape[0] == 0:
                    break
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                l[i] = rgb
            #cam.set(cv2.CAP_PROP_POS_FRAMES, cam.get(cv2.CAP_PROP_POS_FRAMES)+180)
            count += 8
            d.update(l, frame_i)
            del(l)
            t2 = time.time()
            if (t2-t1) >= 1:
                print(count/(t2-t1))
                print(len(d.all_faces), len(d.all_poses))
                count = 0
                t1=t2
    except KeyboardInterrupt:
    #except:
        pass
    print("start cluster")
    d.cluster()
    with open(input_file + ".poses", "wb") as f:
        pickle.dump(d.pose_by_label, f)




