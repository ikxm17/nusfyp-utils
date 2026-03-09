# Notebooks

Jupyter notebooks for post-processing and visualization of nerfstudio outputs.

## frames_to_video.ipynb

Converts sequential image frames from nerfstudio renders into shareable MP4 videos.

### Purpose

Nerfstudio renders output individual frames as image files. This notebook stitches them together into a video for sharing and presentation.

### Usage

1. Set the configuration variables in the notebook:
   - `INPUT_FOLDER` — directory containing image frames
   - `OUTPUT_FILE` — path for the output video
   - `FPS` — frames per second (default: 30)
   - `CODEC` — video codec (default: `mp4v`)
2. Run all cells

### Features

- Supports multiple image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`
- Natural sort order so `frame2` comes before `frame10`
- Auto-detects dimensions from the first frame
- Resizes mismatched frames automatically
- Progress indicator during processing
- Optional in-notebook video playback

### Dependencies

- `opencv-python` (`cv2`)
- `IPython.display` (for in-notebook playback)
