import os

file_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.abspath(os.path.join(file_path, os.pardir, os.pardir))

# Map classes to integers
label_dict = {
    "HAPPY": 0,
    "SAD": 1,
    "ANGRY": 2,
    "FEAR": 3,
    "SURPRISE": 4,
    "HATE": 5,
    "OTHER": 6
}
