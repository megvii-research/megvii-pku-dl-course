import cv2
import argparse
import os
import numpy as np
from PIL import Image
import random
import colorsys

import torch
import torch.nn as nn
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms

from onnxruntime_extensions import onnx_op, PyCustomOpDef, PyOrtFunction


INPUT_FORMAT = "rgb"
TEST_IMAGE_SIZE = (608, 608)
BACKBONE_PATH = None
OUT_FEATURES = ["dark3", "dark4", "dark5"]

THING_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "trafficlight",
    "firehydrant",
    "stopsign",
    "parkingmeter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sportsball",
    "kite",
    "baseballbat",
    "baseballglove",
    "skateboard",
    "surfboard",
    "tennisracket",
    "bottle",
    "wineglass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hotdog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cellphone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddybear",
    "hairdrier",
    "toothbrush",
]
NUM_CLASSES = len(THING_CLASSES)

ANCHORS = [
    [[116, 90], [156, 198], [373, 326]],
    [[30, 61], [62, 45], [42, 119]],
    [[10, 13], [16, 30], [33, 23]],
]
NUM_ANCHORS = len(ANCHORS)

CONF_THRESHOLD = 0.01
NMS_THRESHOLD = 0.5
IGNORE_THRESHOLD = 0.7


from onnxruntime_extensions import onnx_op, PyCustomOpDef, PyOrtFunction


def load_onnx(model_path):
    @onnx_op(
        op_type="Volcano::Quant",
        inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_int64],
        attrs=["dtype"],
    )
    def Quant(x, scale, zero_point, dtype):
        if dtype == "u16":
            min_value = 0
            max_value = 65535
        elif dtype == "u8":
            min_value = 0
            max_value = 255
        elif dtype == "u4":
            min_value = 0
            max_value = 15
        elif dtype == "u2":
            min_value = 0
            max_value = 3

        x = x / scale + zero_point
        x = np.clip(x, min_value, max_value)
        x = np.floor(x + 0.5)
        return (x - zero_point) * scale

    return PyOrtFunction.from_model(model_path)


def decode_predictions(
    input, anchors, image_size, num_classes, num_anchors, is_train=False
):
    bs = input.size(0)  # batch_size
    in_h = input.size(2)  # input_height
    in_w = input.size(3)  # input_weight
    stride_h = image_size[1] / in_h
    stride_w = image_size[0] / in_w
    bbox_attrs = 1 + 4 + num_classes
    prediction = (
        input.view(bs, num_anchors, bbox_attrs, in_h, in_w)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )

    # Get outputs
    scaled_anchors = [(a_w, a_h) for a_w, a_h in anchors]
    x = torch.sigmoid(prediction[..., 0])  # Center x
    y = torch.sigmoid(prediction[..., 1])  # Center y
    w = prediction[..., 2]  # Width
    h = prediction[..., 3]  # Height
    conf = torch.sigmoid(prediction[..., 4])  # Conf
    pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    # Calculate offsets for each grid
    grid_x = (
        torch.linspace(0, in_w - 1, in_w)
        .repeat(in_h, 1)
        .repeat(bs * num_anchors, 1, 1)
        .view(x.shape)
        .type(FloatTensor)
    )
    grid_y = (
        torch.linspace(0, in_h - 1, in_h)
        .repeat(in_w, 1)
        .t()
        .repeat(bs * num_anchors, 1, 1)
        .view(y.shape)
        .type(FloatTensor)
    )
    # Calculate anchor w, h
    anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
    anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
    anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
    # Add offset and scale with anchors
    pred_boxes = prediction[..., :4].clone()
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
    pred_boxes[..., 0] *= stride_w
    pred_boxes[..., 1] *= stride_h
    pred_boxes = pred_boxes.data
    if is_train:
        return conf, x, y, w, h, pred_cls, pred_boxes
    else:
        output = torch.cat(
            (
                pred_boxes.view(bs, -1, 4),
                conf.view(bs, -1, 1),
                pred_cls.view(bs, -1, num_classes),
            ),
            -1,
        )
        return output


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def generalized_batched_nms(
    boxes, scores, idxs, iou_threshold, score_threshold=0.001, nms_type="normal"
):
    assert boxes.shape[-1] == 4

    if nms_type == "normal":
        keep = batched_nms(boxes, scores, idxs, iou_threshold)
    else:
        raise NotImplementedError('NMS type not implemented: "{}"'.format(nms_type))

    return keep


def postprocess(
    prediction, num_classes, conf_thre=0.7, nms_thre=0.5, nms_type="normal"
):
    """
    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 4)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h` where `xc` and `yc` represent a center
            of a bounding box.
        num_classes (int):
            number of dataset classes.
        conf_thre (float):
            confidence threshold ranging from 0 to 1,
            which is defined in the config file.
        nms_thre (float):
            IoU threshold of non-max suppression ranging from 0 to 1.

    Returns:
        output (list of torch tensor):

    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        confidence = detections[:, 4] * detections[:, 5]
        nms_out_index = generalized_batched_nms(
            detections[:, :4],
            confidence,
            detections[:, -1],
            nms_thre,
            nms_type=nms_type,
        )
        detections[:, 4] = confidence / detections[:, 5]

        detections = detections[nms_out_index]
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].unique()

        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            if output[i] is None:
                output[i] = detections_class
            else:
                output[i] = torch.cat((output[i], detections_class))
    return output


def postprocess_boxes(pred_bbox, src_size, eval_size):
    pred_coor = pred_bbox
    src_w, src_h = src_size
    eval_w, eval_h = eval_size
    resize_ratio_w = float(eval_w) / src_w
    resize_ratio_h = float(eval_h) / src_h
    dw = (eval_size[0] - resize_ratio_w * src_w) / 2
    dh = (eval_size[1] - resize_ratio_h * src_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio_w
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio_h
    pred_coor = np.concatenate(
        [
            np.maximum(pred_coor[:, :2], [0, 0]),
            np.minimum(pred_coor[:, 2:], [src_w - 1, src_h - 1]),
        ],
        axis=-1,
    )
    return pred_coor


def draw_bbox(
    src_image, bboxes, scores, classe_inds, classes, show_label=True, input_format="rgb"
):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    image = src_image.copy()
    if input_format.lower() == "rgb":
        image = image[:, :, ::-1]
    image = image.astype(np.uint8)

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors)
    )

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = 1 if scores is None else scores[i]
        class_ind = int(classe_inds[i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = "%s: %.2f" % (classes[class_ind], score)
            t_size = cv2.getTextSize(
                bbox_mess, 0, fontScale, thickness=bbox_thick // 2
            )[0]
            cv2.rectangle(
                image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1
            )  # filled

            cv2.putText(
                image,
                bbox_mess,
                (c1[0], c1[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                (0, 0, 0),
                bbox_thick // 2,
                lineType=cv2.LINE_AA,
            )

    return image


def get_parser():
    parser = argparse.ArgumentParser(description="inference code for builtin models")
    parser.add_argument("--model-path", default="yolov3_8w8f.onnx", help="")
    parser.add_argument("--image-path", default="000000000139.jpg", help="input images")
    parser.add_argument("--output-dir", default="inference_output", help="")

    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    input_format = INPUT_FORMAT

    conf_threshold = CONF_THRESHOLD
    nms_threshold = NMS_THRESHOLD

    original_image = np.asarray(Image.open(args.image_path).convert("RGB"))  # rgb

    if input_format.lower() == "bgr":
        # whether the model expects BGR inputs or RGB
        original_image = original_image[:, :, ::-1]
    height, width = original_image.shape[:2]

    image = original_image.copy()
    pil_image = Image.fromarray(image)
    interp_method = Image.BILINEAR
    pil_image = pil_image.resize(TEST_IMAGE_SIZE, interp_method)
    image = np.asarray(pil_image)
    image = image.transpose(2, 0, 1)
    img_size = image.shape[-2:]

    model = load_onnx(args.model_path)

    img_encoded = image / 255.0
    img_encoded = img_encoded[None, :, :, :].astype("float32")

    outputs = model(img_encoded)

    predictions = [
        decode_predictions(
            torch.from_numpy(out),
            a,
            image_size=TEST_IMAGE_SIZE,
            num_classes=NUM_CLASSES,
            num_anchors=NUM_ANCHORS,
            is_train=False,
        )
        for out, a in zip(outputs, ANCHORS)
    ]
    predictions = torch.cat(predictions, 1)
    detections = postprocess(predictions, NUM_CLASSES, conf_threshold, nms_threshold)

    detection = detections[0].cpu().detach().numpy()
    boxes = postprocess_boxes(
        detection[:, :4], src_size=(width, height), eval_size=TEST_IMAGE_SIZE
    )

    scores = detection[:, 5] * detection[:, 4]
    classe_inds = detection[:, -1]

    image_with_box = draw_bbox(
        original_image,
        boxes,
        scores,
        classe_inds,
        THING_CLASSES,
        input_format=input_format,
    )
    if args.output_dir:
        out_filename = os.path.join(args.output_dir, os.path.basename(args.image_path))
        cv2.imwrite(out_filename, image_with_box)
