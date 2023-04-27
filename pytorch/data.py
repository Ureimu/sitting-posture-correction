import json

file = open(r"../openposeOutput/json/COCO_val2014_000000000192_keypoints.json", mode='r')
data = json.load(file)
print(data["people"][0]["pose_keypoints_2d"])