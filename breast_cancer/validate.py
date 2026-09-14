
def is_valid_input(
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean
):

    if not (0 < radius_mean < 40):
        return False

    if not (0 < texture_mean < 50):
        return False

    if not (0 < perimeter_mean < 250):
        return False

    if not (0 < area_mean < 3000):
        return False

    return True
