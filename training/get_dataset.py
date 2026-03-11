from roboflow import Roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="xU5K9pp8upqFLYLfirA0")
project = rf.workspace("lacto13acillus").project("penalty-kick")
version = project.version(2)
dataset = version.download("yolov8")
                
                
                
                