def dict_from_attribute(inst_dict, attr, transform=lambda x: x):
    return {
        inst_id: transform(getattr(inst, attr))
        for inst_id, inst in inst_dict.items()
    }