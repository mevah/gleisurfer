#!/usr/bin/env python
import os
import sys
import cv2
import datetime
import numpy as np
import torch

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from rail_marking.segmentation.deploy import RailtrackSegmentationHandler
    from cfg import BiSeNetV2Config
    from scripts.segmentation.test_one_image import postprocess_seg, overlay_detections
except Exception as e:
    print(e)
    sys.exit(0)


def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    '''Resize named cv2 window while keeping the aspect ratio.'''
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None or ((h / w) >= (height / width)):
        r = height / float(h)
        dim = (int(w * r), height)
    if height is None or ((h / w) < (height / width)):
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, required=True)
    parser.add_argument("-video_path", type=str, required=True)
    parser.add_argument("-output_video_path", type=str, default="result.mp4")
    parser.add_argument("-skip_every", type=int, default=5)

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    segmentation_handler = RailtrackSegmentationHandler(args.snapshot, BiSeNetV2Config())

    capture = cv2.VideoCapture(args.video_path)
    if not capture.isOpened():
        raise Exception("failed to open {}".format(args.video_path))
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length, "frames")

    width = int(capture.get(3))
    height = int(capture.get(4))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    out_video = cv2.VideoWriter(args.output_video_path, fourcc, fps, (width, height))

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

    _total_ms = 0
    count_frame = 0
    while capture.isOpened():
        ret, frame = capture.read()
        count_frame += 1

        if count_frame % args.skip_every != 0:
            continue

        if not ret:
            break

        start = datetime.datetime.now()
        mask, overlay = segmentation_handler.run(frame, only_mask=False)
        _total_ms += (datetime.datetime.now() - start).total_seconds() * 1000
        new_mask, new_overlay = postprocess_seg(segmentation_handler, mask, frame)
        new_overlay = overlay_detections(model, frame, new_overlay)
        out_video.write(new_overlay)

        new_overlay = resizeWithAspectRatio(new_overlay, width=1920 // 2, height=1080 // 2, inter=cv2.INTER_AREA)
        cv2.imshow("result", np.hstack([new_overlay]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("processing time one frame {}[ms]".format(_total_ms / count_frame))

    capture.release()
    out_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
