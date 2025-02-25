# Vehicle Detection and Tracking using Heatmap

This project demonstrates vehicle detection using YOLOv8 and heatmaps. The script processes a video, detects vehicles, and annotates the frames with bounding boxes, labels, traces, and heatmaps.

## Features

- **YOLOv8**: Utilizes YOLOv8 for vehicle detection.
- **Supervision**: Incorporates supervision tools for annotations.
- **Heatmaps**: Generates heatmaps to visualize the detection activity.
- **Real-time Annotations**: Adds bounding boxes, labels, traces, and heatmaps to video frames.
- **Timestamp**: Displays the current date and time on each frame.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/zaid-kamil/vduhm.git
   cd vduhm
   ```

2. Install PyTorch with CPU-only support:
   ```sh
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Test the heatmap:
   ```sh
   python heatmap_det.py
   ```

2. Run the main script:
   ```sh
   python tracker.py
   ```

## Dependencies

- OpenCV
- NumPy
- Supervision
- YOLO

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [YOLO](https://github.com/ultralytics/yolov5)
- [Supervision](https://github.com/your-supervision-repo)
