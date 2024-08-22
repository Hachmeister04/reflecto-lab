def round_to_nearest(value, value_list):
    return min(value_list, key=lambda x: abs(x - value))