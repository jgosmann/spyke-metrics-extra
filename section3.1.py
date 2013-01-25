#!/usr/bin/env python

from __future__ import division
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

trains = {}
test_trains = {}
for rate in sp.linspace(1 * pq.Hz, max_rate, evaluation_points):
    trains[rate] = [stg.gen_homogeneous_poisson(
        rate, t_stop=spike_train_length) for i in xrange(spike_trains_per_rate)]
    test_trains[rate] = stg.gen_homogeneous_poisson(
        rate, t_stop=spike_train_length)

results = {}
for tau in time_scales:
    results[tau] = sp.empty((evaluation_points, evaluation_points))
    for i, ri in enumerate(trains.iterkeys()):
        for j, rj in enumerate(trains.iterkeys()):
            print i, j
            results[tau][i, j] = sp.mean([stm.victor_purpura_dist(
                [st, test_trains[rj]], 2.0 / tau, sort=False)
                for st in trains[ri]])
    plt.figure()
    plt.title(str(tau))
    plt.imshow(results[tau])
    plt.colorbar()
#plt.show()
