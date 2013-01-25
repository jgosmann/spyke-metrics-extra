#!/usr/bin/env python

from __future__ import division
import json
import matplotlib.pyplot as plt
import os.path
import pickle
import quantities as pq
import scipy as sp
import spykeutils.spike_train_generation as stg
import spykeutils.spike_train_metrics as stm
import time

with open("conf/testing.conf") as config_file:
    config = json.load(config_file)
evaluation_points = config['evaluation_points']
max_rate = pq.Quantity(*config['max_rate'])
spike_trains_per_rate = config['spike_trains_per_rate']
spike_train_length = pq.Quantity(*config['spike_train_length'])
#time_scales = sp.array([10, 50, 100, 500, 1000, 5000]) * pq.ms
time_scales = pq.Quantity(*config['time_scales'])


class Task(object):
    def __init__(self, target, dependencies=[], name=None):
        self.target = target
        self.dependencies = dependencies
        if name is None:
            self.name = target.__name__
        else:
            self.name = name
        self.is_built = False

    def last_build_time(self):
        return 0

    def needs_rebuild(self):
        timestamp = self.last_build_time()
        if len(self.dependencies) <= 0:
            return timestamp <= 0
        else:
            return any((d.last_build_time() > timestamp
                        for d in self.dependencies))

    def build(self):
        for d in self.dependencies:
            d.build()
        if self.needs_rebuild():
            self.result = self.target(*[d.result for d in self.dependencies])
            self.is_built = True


class PersistingTask(Task):
    def __init__(self, target, storage_dir, dependencies=[], name=None):
        Task.__init__(self, target, dependencies, name)
        self.storage_dir = storage_dir

    def storage_path(self):
        return os.path.join(self.storage_dir, self.name)

    def store(self):
        with open(self.storage_path(), 'w') as storage_file:
            pickle.dump(self.result, storage_file)

    def load(self):
        with open(self.storage_path()) as storage_file:
            self.result = pickle.load(storage_file)

    def last_build_time(self):
        try:
            return os.path.getmtime(self.storage_path())
        except:
            return 0

    def build(self):
        for d in self.dependencies:
            d.build()
        if self.needs_rebuild():
            print "Building:", self.name
            self.result = self.target(*[d.result for d in self.dependencies])
            self.is_built = True
            self.store()
        elif not self.is_built:
            time_str = time.strftime('%x %X', time.localtime(self.last_build_time()))
            print "Up to date: %s (%s)" % (self.name, time_str)
            self.load()
            self.is_built = True


def gen_trains():
    trains = {}
    test_trains = {}
    for rate in sp.linspace(1 * pq.Hz, max_rate, evaluation_points):
        trains[rate] = [stg.gen_homogeneous_poisson(
            rate, t_stop=spike_train_length) for i in xrange(spike_trains_per_rate)]
        test_trains[rate] = stg.gen_homogeneous_poisson(
            rate, t_stop=spike_train_length)
    return trains, test_trains


def calc_metrics((trains, test_trains)):
    results = {}
    for tau in time_scales:
        results[tau] = sp.empty((evaluation_points, evaluation_points))
        for i, ri in enumerate(trains.iterkeys()):
            for j, rj in enumerate(trains.iterkeys()):
                print i, j
                results[tau][i, j] = sp.mean([stm.victor_purpura_dist(
                    [st, test_trains[rj]], 2.0 / tau, sort=False)
                    for st in trains[ri]])
    return results


def plot(results):
    for tau in time_scales:
        plt.figure()
        plt.title(str(tau))
        plt.imshow(results[tau])
        plt.colorbar()
        plt.show()

t1 = PersistingTask(gen_trains, 'data')
t2 = PersistingTask(calc_metrics, 'data', [t1])
t3 = PersistingTask(plot, 'data', [t2])

t3.build()
