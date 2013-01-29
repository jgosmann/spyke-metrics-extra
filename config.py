
import json
import quantities as pq


class ConfigType(object):
    def unpack(self, json):
        raise NotImplementedError()


class Integer(ConfigType):
    def unpack(self, json):
        return int(json)


class String(ConfigType):
    def unpack(self, json):
        return str(json)


class List(ConfigType):
    def unpack(self, json):
        return list(json)


class Quantity(ConfigType):
    def __init__(self, units=None):
        self.units = units

    def unpack(self, json):
        unpacked = pq.Quantity(*json).astype(float)
        if self.units is not None:
            unpacked.units = self.units
        return unpacked


class ConfigList(ConfigType):
    def __init__(self, specification):
        self.spec = ConfigSpec(specification)

    def unpack(self, json):
        return [self.spec.unpack(cfg) for cfg in json]


class ConfigSpec(ConfigType):
    def __init__(self, specification):
        self.specification = specification

    def unpack(self, json):
        config = {}
        for key, spec in self.specification.iteritems():
            config[key] = spec.unpack(json[key])
        return config


def load(specification, file_obj):
    return specification.unpack(json.load(file_obj))
