import os

def round_to_nearest(value: float, value_list: list):
    return min(value_list, key=lambda x: abs(x - value))

def get_shot_from_path(path: str):
    try:
        return int(os.path.basename(path))
    except ValueError:
        raise Exception("Wrong file name")

def find_path_from_shot(shot: int):
    return f"/shares/experiments/aug-rawfiles/RTR/{shot//10}/{shot}"