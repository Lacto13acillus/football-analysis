from roboflow import Roboflow
rf = Roboflow(api_key="xU5K9pp8upqFLYLfirA0")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(20)
dataset = version.download("yolov5")
                