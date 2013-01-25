#!/usr/bin/env python

from __future__ import division
from joblib import Memory
import config
import itertools
import logging
import matplotlib.pyplot as plt
import quantities as pq
import scipy as sp
import spykeutils.spike_train_generation as stg
import spykeutils.spike_train_metrics as stm

name = 'section3.1'
memory = Memory(cachedir='cache')
logger = logging.getLogger(name)

config_spec = config.ConfigSpec({
    name: config.ConfigSpec({
        'evaluation_points': config.Integer(),
        'max_rate': config.Quantity(pq.Hz),
        'spike_trains_per_rate': config.Integer(),
        'spike_train_length': config.Quantity(pq.s),
        'time_scales': config.Quantity(pq.s)
    })
})

with open("conf/testing.conf") as config_file:
    cfg = config.load(config_spec, config_file)[name]

rates = sp.linspace(1 * pq.Hz, cfg['max_rate'], cfg['evaluation_points'])


@memory.cache
def gen_trains():
    logger.info("Generating spike trains")
    trains_by_rate = []
    for rate in rates:
        trains_by_rate.append([stg.gen_homogeneous_poisson(
            rate, t_stop=cfg['spike_train_length'])
            for i in xrange(cfg['spike_trains_per_rate'])])
    return tuple(trains_by_rate)


@memory.cache
def calc_metric(trains_by_rate):
    logger.info("Calculating Metric")
    results_by_time_scale = sp.empty((
        cfg['time_scales'].size, cfg['evaluation_points'],
        cfg['evaluation_points']))
    for t, tau in enumerate(cfg['time_scales']):
        logger.info("  for time scale %s" % str(tau))
        dist_mat = stm.victor_purpura_dist(
            list(itertools.chain(*trains_by_rate)), 2.0 / tau, sort=False)
        for i in xrange(cfg['evaluation_points']):
            i_start = cfg['spike_trains_per_rate'] * i
            i_stop = i_start + cfg['spike_trains_per_rate']
            for j in xrange(i, cfg['evaluation_points']):
                j_start = cfg['spike_trains_per_rate'] * j
                j_stop = j_start + cfg['spike_trains_per_rate']
                results_by_time_scale[t, i, j] = \
                    results_by_time_scale[t, j, i] = sp.mean(dist_mat[
                        i_start:i_stop, j_start:j_stop])
    return results_by_time_scale


def plot(results):
    for t, tau in enumerate(cfg['time_scales']):
        plt.figure()
        plt.title(str(tau))
        plt.imshow(results[t])
        plt.colorbar()
        plt.show()


logger.info("Section 3.1")
plot(calc_metric(gen_trains()))
