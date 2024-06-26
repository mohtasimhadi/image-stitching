# Panoramic Image Stitching

## Camera Configuration For the Test Images

**OAK-D Pro**


| Spec                   | Color                       | Stereo             |
| ---------------------- | --------------------------- | ------------------ |
| Sensor                 | IMX378                      | OV9282             |
| DFOV/HFOV/VFOV         | 81° / 69° / 55°          | 89° / 80° / 55° |
| Resolution             | 12 MP (4056*3040)           | 1MP (1280*800)     |
| Focus                  | AF: 8cm - ∞, FF: 50cm - ∞ | FF: 19.6cm - ∞    |
| Max Framerate          | 60 FPS                      | 120 FPS            |
| F-number               | 1.8 ± 5%                   | 2.0 ± 5%          |
| Lens Size              | 1/2.3 inch                  | 1/4 inch           |
| Effective Focal Length | 4.81 mm                     | 2.35 mm            |
| Pixel Size             | 1.55µm * 1.55µm           | 3.0 µm * 3.00 µm |

Full camera documentation can be found [here.](https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9098pro/)

## Installation

```bash
git clone <repository link>
cd <repository directory>
pip install -r requirements.txt
```

## Usage

### Extracting Frames from Videos

```bash
python main.py extract_frames <video_path> <output_path> <rgb/mono> <fps>
```

### Invariant Features

```bash
python main.py <input directory> <output directory> <batch size>
```
