from roboflow import Roboflow
rf = Roboflow(api_key="KvsGUXdoY9TgVZjYJtx2")
project = rf.workspace("falldetection-bwfdz").project("yolo-dataset-98swd")
version = project.version(2)
dataset = version.download("yolov11")
                