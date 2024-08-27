def round_to_nearest(value: float, value_list: list):
    return min(value_list, key=lambda x: abs(x - value))

def shot(path: str):
    try:
        return int(path.split('\\')[-1])
    except ValueError:
        raise Exception("Wrong file name")

def find_path_from_shot(shot: int):
    return f"/shares/experiments/rawfiles/RTR/{shot//10}/{shot}"