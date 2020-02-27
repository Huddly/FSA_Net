import cv2
import sys
sys.path.append('..')
from math import cos, sin
from lib.FSANET_model import *
import numpy as np
from keras.layers import Average
from keras.models import Model

from bbox import AnnotationContainer, AnnotationEntry, AnnotationInstance, BBox
from tqdm import tqdm


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50):
    print(yaw,roll,pitch)
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(255,0,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(0,0,255),2)

    return img


def get_orientation_model(input_side=64):
    # Parameters
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3, 3, 3]
    lambda_d = 1
    num_classes = 3
    image_size = input_side
    num_primcaps = 7 * 3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    num_primcaps = 8 * 8 * 3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    print('Loading models ...')

    weight_file1 = '../pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')

    weight_file2 = '../pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')

    weight_file3 = '../pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')

    inputs = Input(shape=(input_side, input_side, 3))
    x1 = model1(inputs)  # 1x1
    x2 = model2(inputs)  # var
    x3 = model3(inputs)  # w/o
    avg_model = Average()([x1, x2, x3])
    model = Model(inputs=inputs, outputs=avg_model)
    return model


def detect_face(image, net, crop_coordinates=None, threshold=0.4):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detected = net.forward()[0, 0, ...]

    conf = detected[:, 2]
    detected = detected[conf > threshold, :]
    detected[:, 3:] = np.clip(detected[:, 3:], a_min=0., a_max=1.)
    detected[:, (3, 5)] *= image.shape[1]
    detected[:, (4, 6)] *= image.shape[0]
    if crop_coordinates is not None:
        detected[:, (3, 5)] += crop_coordinates[0]
        detected[:, (4, 6)] += crop_coordinates[1]

    faces = []
    for f in detected:
        coor = f[3:].astype(int)
        if coor[0] >= coor[2] or coor[1] >= coor[3]:
            continue
        faces.append(
            AnnotationInstance(
                bbox=BBox(
                    xmin=coor[0], ymin=coor[1], xmax=coor[2], ymax=coor[3],
                    label='face', score=f[2], coordinate_mode='absolute'
                )
            )
        )

    return faces


def infer_head_orientation(image, model: Model):
    input_shape = model.input_shape[1:3]

    image = cv2.resize(image, input_shape)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    face = np.expand_dims(image, axis=0)

    p_result = model.predict(face)
    return p_result[0]


def add_face_and_head_orientation(
        entry: AnnotationEntry,
        frame: np.ndarray,
        facenet,
        orientation_model: Model,
        padding_factor: float = 0.3,
        threshold: float = 0.4,
        nms_threshold: float = 0.7
):

    def crop_image(i: AnnotationInstance):
        coor_mode = i.bbox.coordinate_mode
        i.bbox.convert_coordinate_mode('absolute', entry.image_size)
        xmin, ymin, xmax, ymax = i.bbox.to_xyxy()
        w_pad, h_pad = (xmax-xmin)*padding_factor, (ymax-ymin)*padding_factor
        xmin, xmax = xmin-w_pad, xmax+w_pad
        ymin, ymax = ymin - h_pad, ymax + h_pad
        xmin, xmax = int(max(0, xmin)), int(min(entry.width, xmax))
        ymin, ymax = int(max(0, ymin)), int(min(entry.width, ymax))

        c = np.array(frame)[ymin:ymax, xmin:xmax, ...]
        i.bbox.convert_coordinate_mode(coor_mode, entry.image_size)
        return c, [xmin, ymin, xmax, ymax]

    # Detecting faces in all head boxes
    # TODO: Replace with better face detector down the line
    new_instances = []
    for i in entry:
        if i.label == 'head':
            crop, crop_coor = crop_image(i)
            faces = detect_face(crop, facenet, crop_coor, threshold=threshold)

            best_face = max(faces, key=lambda f: f.score) if len(faces) > 0 else None

            # TODO: adding highest conf detection as this heads face. (may cause problems in  the future)
            if best_face is not None:
                best_face.meta['tracking_id'] = i.meta['tracking_id']
                new_instances.append(best_face)

            # new_instances += faces
    for f in new_instances:
        entry.add_instance(f)

    # TODO: If multiple faces is detected within each crop: This will remove overlapping boxes at least
    # entry.apply_non_maximum_suppression(nms_threshold, inplace=True)

    # Detect head orientation as yaw/pitch/roll and add those values to face meta
    for i in entry:
        if i.label == 'face':
            crop, _ = crop_image(i)
            yaw, pitch, roll = infer_head_orientation(crop, orientation_model)
            i.meta['yaw'] = float(yaw)
            i.meta['pitch'] = float(pitch)
            i.meta['roll'] = float(roll)


if __name__ == '__main__':
    # main()
    def get_frame_generator(video_path):
        vidcap = cv2.VideoCapture(str(video_path))
        frame = 0
        while True:
            success, image = vidcap.read()
            if success: # and frame < 500:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yield frame, image
                frame += 1
            else:
                break

    from matplotlib import pyplot as plt

    input_side = 64
    facenet = cv2.dnn.readNetFromCaffe(
        '../demo/face_detector/deploy.prototxt',
        '../demo/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    )
    orientation_model = get_orientation_model(input_side=input_side)

    container = AnnotationContainer.from_file('/home/martin/Desktop/meeting_cut/spaces/no_crop/task_nr_354.bbox')
    for fnr, frame in tqdm(get_frame_generator('/home/martin/Desktop/meeting_cut/spaces/no_crop/overview.mp4'),
                           total=5483):
        entry = container.entries[str(fnr)]
        add_face_and_head_orientation(entry, frame, facenet, orientation_model)

        # img = entry.overlaid_on_image(frame)
        # plt.imshow(img)
        # plt.show()

    container.to_file('/home/martin/Desktop/meeting_cut/spaces/no_crop/task_nr_354_face_and_orientation.bbox')