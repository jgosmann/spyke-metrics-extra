
import json
import quantities as pq
import scipy as sp


class Quantity(object):
    def __init__(self, units=None):
        self.units = units

    def __call__(self, json):
        unpacked = pq.Quantity(*json).astype(float)
        if self.units is not None:
            unpacked.units = self.units
        return unpacked


class QuantityLogRange(object):
    def __init__(self, units=None):
        self.units = units

    def __call__(self, json):
        unpacked = pq.Quantity(sp.logspace(*json[0]), json[1])
        if self.units is not None:
            unpacked.units = self.units
        return unpacked


class ConfigList(object):
    def __init__(self, specification):
        self.spec = ConfigSpec(specification)

    def __call__(self, json):
        return [self.spec(cfg) for cfg in json]


class ConfigSpec(object):
    def __init__(self, specification):
        self.specification = specification

    def __call__(self, json):
        config = {}
        for key, spec in self.specification.iteritems():
            config[key] = spec(json[key])
        return config


def load(specification, file_obj):
    return specification(json.load(file_obj))
