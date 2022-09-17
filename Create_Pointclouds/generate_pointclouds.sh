path_template_config_file = "path/to/gleisurfer/Create_Pointclouds/config.yaml"
input_video_location="path/to/video"
output_sfm_location="path/to/output/sfm/folder"

mkdir $output_sfm_location

cp -R $path_template_config_file $output_sfm_location
ffmpeg -i input_video_location.mp4 -vf fps=3 "$output_sfm_location/images"+%d.png
bin/opensfm_run_all $output_sfm_location