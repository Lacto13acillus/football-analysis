from roboflow import Roboflow
rf = Roboflow(api_key="xU5K9pp8upqFLYLfirA0")
project = rf.workspace("lacto13acillus").project("dribbling-football")
version = project.version(1)
dataset = version.download("yolov8")
                