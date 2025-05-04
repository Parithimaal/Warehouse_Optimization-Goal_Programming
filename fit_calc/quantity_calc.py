import math

def qty_base_layer_factory(*, dimensions, bins):
    """\
    Computes the number of parts of part p that will fit on the floor of the bin in 2 orientations
    """
    def qty_base_layer(p, b):
        length_p, width_p = dimensions[p].length, dimensions[p].width
        length_b, width_b = bins[b].length, bins[b].width
        fit_1 = math.floor(length_b/length_p) * math.floor(width_b/width_p)
        fit_2 = math.floor(length_b/width_p) * math.floor(width_b/length_p)
        return max(fit_1, fit_2)
    return qty_base_layer

def qty_max_factory(*, dimensions, bins, is_stackable_p_, qty_base_layer):
    """\
    Computes the max quantity of part p in bin b
    """
    def qty_max(p, b):
        base = qty_base_layer(p,b)
        if base == 0:
            return 0
        # Calculating number of layers
        if is_stackable_p_[p]==0:
            layers = 1
        else:
            layers = math.floor(bins[b].height / dimensions[p].height)
        return base*layers
    return qty_max

