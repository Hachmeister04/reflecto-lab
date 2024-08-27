def round_to_nearest(value, value_list):
    return min(value_list, key=lambda x: abs(x - value))

def shot(path):
    try:
        return int(path.split('\\')[-1])
    except ValueError:
        raise Exception("Wrong file name")