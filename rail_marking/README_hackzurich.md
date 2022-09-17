# Rail marking and clearance overlay

## References

This code use the [rail_marking](https://github.com/xmba15/rail_marking) repo with a few changes in `test_video.py` and `test_one_image.py` under `scripts/segmentation`. It also uses [yolov5](https://github.com/ultralytics/yolov5) for object detection.

## Environment

```bash
conda create --name gleisurfer python=3.8
conda activate gleisurfer
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge brotlipy
pip install matplotlib requests albumentations==0.4.3 nudged tqdm pandas seaborn psutil
```

## Usage

[Pretrained segmentation model](https://drive.google.com/file/d/11FAmJR79bmO0SjzQIqBvWD8Zy9MTWYw2/view?usp=sharing) needs to be downloaded in the folder.

Here is an example usage of the script with one of the sample videos
```bash
python scripts/segmentation/test_video.py -snapshot bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth -video_path /path_to_videos/Videos_with_sensordata/2022_08_17_14_00_27/movie.mp4
```
