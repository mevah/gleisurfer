#!/usr/bin/env python
import os
import sys
import cv2
import datetime
import numpy as np
import math
from types import SimpleNamespace
from PIL import Image
from skimage.transform import ProjectiveTransform
import nudged
import torch

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from rail_marking.segmentation.deploy import RailtrackSegmentationHandler
    from cfg import BiSeNetV2Config
except Exception as e:
    print(e)
    sys.exit(0)


clearance_clr = list((255, 215, 50))[::-1]
clearance_diagram = cv2.imread("clearance_diagram.png", 0)
clearance_diagram = 255 - clearance_diagram
clearance_base_coords = [[102, 600], [358, 596]]


def resize_mask(mask, w, h):
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.resize((w, h), Image.NEAREST)
    return np.array(mask)


def connected_components(mask, ids=[0, 1]):
    comps = []
    if ids is None:
        ids = np.unique(mask)
    for label in ids:
        binary_map = (mask == label).astype(np.uint8)
        output = cv2.connectedComponentsWithStats(binary_map, 4, cv2.CV_32S)
        num_labels, labels, stats, centroids = output
        for idx, stat in enumerate(stats):
            if idx > 0:
                x, y, w, h, a = stat
                comp = SimpleNamespace(
                    label=label,
                    area=a,
                    mask=labels == idx,
                    centroid=centroids[idx],
                )
                comps.append(comp)
    return comps


def get_comp_by_ids(comps, ids, sort_area=True):
    grp = [comp for comp in comps if comp.label in ids]
    if sort_area:
        grp.sort(key=lambda x: x.area, reverse=True)
    return grp


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, required=True)
    parser.add_argument("-image_path", type=str, required=True)
    parser.add_argument("-output_image_path", type=str, default="result.png")
    parser.add_argument("-num_test", type=int, default=1)

    args = parser.parse_args()

    return args


def postprocess_seg(segmentation_handler, mask, image):
    """ 0:rails, 1:track, 2:background """
    mask_height = mask.shape[0]
    mask_width = mask.shape[1]

    # delete spurious segmentations using connected-components
    comps = connected_components(mask, [0, 1])
    rails = get_comp_by_ids(comps, [0])
    track = get_comp_by_ids(comps, [1])
    
    # delete segmentations with area less than half of the maximal one
    rails = [x for x in rails if x.area > rails[0].area // 2]
    track = [x for x in track if x.area > track[0].area // 2]
    new_mask = np.ones_like(mask) * 2
    for t in track:
        new_mask[t.mask] = 1
    for r in rails:
        new_mask[r.mask] = 0

    # delete erroneous track detections outisde rails (assume vertical pose)
    cutoff = 0
    for i in range(mask_height):
        row = new_mask[i, :]
        if 0 in row:
            before = row.tolist().count(1)
            left = row.tolist().index(0)
            right = len(row) - row[::-1].tolist().index(0)
            if 1 in row[left:right]:
                row[:left] = 2
                row[right:] = 2
    #         elif left < mask_width // 2:
    #             row[:left] = 2
    #         else:
    #             row[right:] = 2
    #         after = row.tolist().count(1)
    #         if before != 0 and after == 0:
    #             cutoff = i
    # if cutoff < mask_height // 2:
    #     new_mask[:cutoff, :] = 2

    # delete unified ends of rails, merges happening at horizon (assume vertical pose)
    for i in range(mask_height):
        row = new_mask[i, :]
        if 0 not in row and 1 not in row:
            continue
        if 0 in row and 1 not in row:
            row[row == 0] = 2
            new_mask[i, :] = row
        if 0 in row and 1 in row:
            if row.tolist().count(0) <= row.tolist().count(1) * 2:
                break
            else:
                row[row == 1] = 2
                row[row == 0] = 2
                new_mask[i, :] = row

    # create an overlay of the new masks
    seg_overlay_alpha = segmentation_handler._overlay_alpha
    orig_height, orig_width = image.shape[:2]
    processed_image = cv2.resize(image, (segmentation_handler._model_config.img_width, segmentation_handler._model_config.img_height))
    new_overlay = np.copy(processed_image)
    color_mask = np.array(segmentation_handler._data_config.RS19_COLORS)[new_mask]
    new_overlay = (((1 - seg_overlay_alpha) * new_overlay) + (seg_overlay_alpha * color_mask)).astype(np.uint8)
    new_overlay = cv2.resize(new_overlay, (orig_width, orig_height))

    # choose base points for the clearance diagram
    resized_mask = resize_mask(new_mask, orig_width, orig_height)
    empty = np.zeros_like(resized_mask)
    # naively place base points a bit further from the camera, assume vertical pose
    rail_base_mask = resized_mask.copy()
    rail_base_mask[:6 * orig_height // 9, :] = 2
    rail_base_mask[7 * orig_height // 9:, :] = 2
    comps = connected_components(rail_base_mask, [0, 1])
    # get largest two rails if there are more than two rail components
    rails = get_comp_by_ids(comps, [0], sort_area=True)[:2]
    base_points = np.array([x.centroid.astype(int) for x in rails])

    # sort base points by x (0: left point, 1: right point)
    rail_base_coord = base_points.tolist()
    rail_base_coord.sort(key=lambda x: x[0])

    # place clearance diagram if both tracks have enough visibility
    if len(base_points) > 1:

        # illustrate base points on the overlay
        cv2.polylines(new_overlay, [base_points], False, (128, 0, 32), thickness=10, lineType=cv2.LINE_8)
        for (x, y) in base_points:
            new_overlay = cv2.circle(new_overlay, (x, y), radius=10, color=(255, 192, 0), thickness=-1)

        # match base coordinates in the frame with those in the clearance diagram
        trans = nudged.estimate(clearance_base_coords, rail_base_coord)
        mat = np.array(trans.get_matrix())
        proj = ProjectiveTransform(matrix=mat, dimensionality=2)
        # transform and fit the diagram to frame (rotating and resizing)
        clearance_trans = cv2.warpAffine(src=clearance_diagram, dst=empty, M=mat[:2,:], dsize=empty.shape[:2][::-1])

        # draw edges of the clearance diagram
        contours, _ = cv2.findContours(clearance_trans, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(new_overlay, [contours[0]], 0, clearance_clr, 3)

        # color inside of the clearance diagram
        clearance_mask = clearance_trans != 0
        new_overlay[clearance_mask] = (np.array(clearance_clr) * 0.2 + 0.8 * new_overlay[clearance_mask]).astype(np.uint8)

    return new_mask, new_overlay


def overlay_detections(model, image, viz):
    # detect objects using yolo and overlay if any non-truck/non-car objects are found
    results = model(image)
    df = results.pandas().xyxy[0]
    for i in range(len(df)):
        name = df.iloc[i]['name']
        if name not in ["car", "truck"]:
            xmin = int(df.iloc[i]['xmin'])
            ymax = int(df.iloc[i]['ymax'])
            ymin = int(df.iloc[i]['ymin'])
            xmax = int(df.iloc[i]['xmax'])
            cv2.rectangle(viz, (xmin, ymin), (xmax, ymax),(0, 255, 0), 2)
    return viz


def main():
    args = get_args()
    segmentation_handler = RailtrackSegmentationHandler(args.snapshot, BiSeNetV2Config())
    image = cv2.imread(args.image_path)
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

    start = datetime.datetime.now()
    for i in range(args.num_test):
        mask, overlay = segmentation_handler.run(image, only_mask=False)
    _processing_time = datetime.datetime.now() - start

    new_mask, new_overlay = postprocess_seg(segmentation_handler, mask, image)
    new_overlay = overlay_detections(model, frame, new_overlay)
    cv2.imwrite(args.output_image_path, new_overlay)

    print("processing time one frame {}[ms]".format(_processing_time.total_seconds() * 1000 / args.num_test))


if __name__ == "__main__":
    main()
