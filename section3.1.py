#!/usr/bin/env python

from __future__ import division
from joblib import Memory
import config
import matplotlib.pyplot as plt
import quantities as pq
import scipy as sp
import spykeutils.spike_train_generation as stg
import spykeutils.spike_train_metrics as stm

config_spec = config.ConfigSpec({'section3.1': config.ConfigSpec({
    'evaluation_points': config.Integer(),
    'max_rate': config.Quantity(pq.Hz),
    'spike_trains_per_rate': config.Integer(),
    'spike_train_length': config.Quantity(pq.s),
    'time_scales': config.Quantity(pq.s)})})

with open("conf/testing.conf") as config_file:
    cfg = config.load(config_spec, config_file)['section3.1']

memory = Memory(cachedir='data')

rates = sp.linspace(1 * pq.Hz, cfg['max_rate'], cfg['evaluation_points'])


@memory.cache
def gen_trains():
    trains = []
    test_trains = []
    for r, rate in enumerate(rates):
        trains.append([stg.gen_homogeneous_poisson(
            rate, t_stop=cfg['spike_train_length']) for i in xrange(cfg['spike_trains_per_rate'])])
        test_trains.append(stg.gen_homogeneous_poisson(
            rate, t_stop=cfg['spike_train_length']))
    return tuple(trains), tuple(test_trains)


@memory.cache
def calc_metrics(trains, test_trains):
    results = []
    for t, tau in enumerate(cfg['time_scales']):
        results.append(sp.empty((cfg['evaluation_points'], cfg['evaluation_points'])))
        for i, ri in enumerate(trains):
            for j, rj in enumerate(trains):
                print i, j
                results[t][i, j] = sp.mean([stm.victor_purpura_dist(
                    [st, test_trains[j]], 2.0 / tau, sort=False)
                    for st in trains[i]])
    return tuple(results)


def plot(results):
    for t, tau in enumerate(cfg['time_scales']):
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
