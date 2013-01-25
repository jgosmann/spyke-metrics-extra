#!/usr/bin/env python

from __future__ import division
from joblib import Memory
import json
import matplotlib.pyplot as plt
import quantities as pq
import scipy as sp
import spykeutils.spike_train_generation as stg
import spykeutils.spike_train_metrics as stm

with open("conf/testing.conf") as config_file:
    config = json.load(config_file)
evaluation_points = config['evaluation_points']
max_rate = pq.Quantity(*config['max_rate'])
spike_trains_per_rate = config['spike_trains_per_rate']
spike_train_length = pq.Quantity(*config['spike_train_length'])
#time_scales = sp.array([10, 50, 100, 500, 1000, 5000]) * pq.ms
time_scales = pq.Quantity(*config['time_scales'])

memory = Memory(cachedir='data')

rates = sp.linspace(1 * pq.Hz, max_rate, evaluation_points)


@memory.cache
def gen_trains():
    trains = []
    test_trains = []
    for r, rate in enumerate(rates):
        trains.append([stg.gen_homogeneous_poisson(
            rate, t_stop=spike_train_length) for i in xrange(spike_trains_per_rate)])
        test_trains.append(stg.gen_homogeneous_poisson(
            rate, t_stop=spike_train_length))
    return tuple(trains), tuple(test_trains)


@memory.cache
def calc_metrics(trains, test_trains):
    results = []
    for t, tau in enumerate(time_scales):
        results.append(sp.empty((evaluation_points, evaluation_points)))
        for i, ri in enumerate(trains):
            for j, rj in enumerate(trains):
                print i, j
                results[t][i, j] = sp.mean([stm.victor_purpura_dist(
                    [st, test_trains[j]], 2.0 / tau, sort=False)
                    for st in trains[i]])
    return tuple(results)


def plot(results):
    for t, tau in enumerate(time_scales):
        plt.figure()
        plt.title(str(tau))
        plt.imshow(results[t])
        plt.colorbar()
        plt.show()

a = gen_trains()
b = calc_metrics(*a)
plot(b)

#t1 = PersistingTask(gen_trains, 'data')
#t2 = PersistingTask(calc_metrics, 'data', [t1])
#t3 = Task(plot, [t2])

#t3.build()
