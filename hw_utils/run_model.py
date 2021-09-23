import os
import sys
import time
import tensorflow as tf
import logging

from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from keras import backend as K
from models.yolov3 import YOLOv3_Darknet53
from models.ssd import SSD512
from PIL import Image
from tog.attacks import *

tf.get_logger().setLevel(logging.ERROR)
K.clear_session()

if len(sys.argv) < 3:
    print("Not enough arguments.", file=sys.stderr)
    exit(-1)

detector = None
n_iter = None
eps = None
eps_iter = None
if sys.argv[1] == 'yolo':
    weights = './weights/YOLOv3_Darknet53.h5'
    detector = YOLOv3_Darknet53(weights=weights)

    eps = 8 / 255.       # Hyperparameter: epsilon in L-inf norm
    eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
    n_iter = 1          # Hyperparameter: number of attack iterations

elif sys.argv[1] == 'ssd':
    weights = './weights/SSD512.h5'
    detector = SSD512(weights=weights)

    eps = 8 / 255.  # Hyperparameter: epsilon in L-inf norm
    eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
    n_iter = 1  # Hyperparameter: number of attack iterations

else:
    print("Invalid detector option", file=sys.stderr)
    exit(-1)

classes_of_interest = ['bird', 'dog', 'person', 'car', 'train']
base_samples_dir = '../assets/Sampled_DS/'

for cls in classes_of_interest:
    save_to_dir = './out/{}/{}/{}'.format(sys.argv[1], sys.argv[2], cls)
    os.makedirs(save_to_dir, exist_ok=True)

    for image in filter(lambda x: x.endswith('.jpg'), os.listdir(base_samples_dir + cls)):

        print("---\nNow processing: {}/{}".format(cls, image))
        try:
            fpath = base_samples_dir + cls + '/' + image
            input_img = Image.open(fpath)

            print("Benign Start: ", time.time())
            x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
            detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)
            visualize_detections({'Benign (No Attack)': (x_query, detections_query,
                                                         detector.model_img_size, detector.classes)},
                                 save_to=save_to_dir + '/' + image.replace('.', '-benign.'))
            print("Benign End: ", time.time())

            save_to = save_to_dir + '/' + image
            if sys.argv[2] == 'vanish':

                print("Attack Start: ", time.time())
                # Generation of the adversarial example
                x_adv_vanishing = tog_vanishing(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)

                detections_adv_vanishing = detector.detect(x_adv_vanishing,
                                                           conf_threshold=detector.confidence_thresh_default)
                visualize_detections({'TOG-vanishing': (x_adv_vanishing, detections_adv_vanishing, detector.model_img_size,
                                                        detector.classes)},
                                     save_to=save_to)
                print("Attack End: ", time.time())

            elif sys.argv[2] == 'mislabel':
                print("Attack Start: ", time.time())
                # Generation of the adversarial examples
                x_adv_mislabeling_ll = tog_mislabeling(victim=detector, x_query=x_query, target='ll', n_iter=n_iter,
                                                       eps=eps, eps_iter=eps_iter)

                detections_adv_mislabeling_ll = detector.detect(x_adv_mislabeling_ll,
                                                                conf_threshold=detector.confidence_thresh_default)
                visualize_detections({'TOG-mislabeling (LL)': (x_adv_mislabeling_ll, detections_adv_mislabeling_ll,
                                                               detector.model_img_size, detector.classes)},
                                     save_to=save_to)
                print("Attack End: ", time.time())
            else:
                print("Invalid attack option", file=sys.stderr)
                exit(-1)

            # print("Done with: {}/{}".format(cls, image))

        except Exception as e:
            print(e)
            print("Exception processing {}/{}".format(cls, image))

