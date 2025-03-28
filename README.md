# Dual Teleop for Astra Arm

See our tutorial on how to use this project:

<https://aha-robot.notion.site/Teleoperation-Handle-Installation-and-Testing-1c033900bc8780008c97cf65191f4bf2>

## Install

```bash
git clone https://github.com/hilookas/astra_teleop.git
cd astra_teleop
pip install -e .
```

## Calibrate the intrinsic parameters of the camera 

```bash
# collect calibration image
python -m astra_teleop.calibration_collect -d /dev/video0 -c ./calibration_images
# process calibration image
python -m astra_teleop.calibration_process -c ./calibration_images
```

## Teleop

```bash
python -m astra_teleop.process -d /dev/video0 -c ./calibration_images
```

## Acknowledgment

Special thanks to Hello Robot team and their stretch dex teleop project!

<https://github.com/hello-robot/stretch_dex_teleop>

## License

[MIT](http://opensource.org/licenses/MIT)

Copyright (c) 2024 Haiqin Cui