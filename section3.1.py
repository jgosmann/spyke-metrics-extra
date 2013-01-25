#!/usr/bin/env python

from __future__ import division
from joblib import Memory
import config
import itertools
import logging
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import quantities as pq
import scipy as sp
import spykeutils.spike_train_generation as stg
import spykeutils.spike_train_metrics as stm

name = 'section3.1'
memory = Memory(cachedir='cache', verbose=0)
logging.basicConfig()
logger = logging.getLogger(name)
logger.setLevel(logging.INFO)

config_spec = config.ConfigSpec({
    name: config.ConfigSpec({
        'evaluation_points': config.Integer(),
        'max_rate': config.Quantity(pq.Hz),
        'spike_trains_per_rate': config.Integer(),
        'spike_train_length': config.Quantity(pq.s),
        'time_scales': config.Quantity(pq.s),
        'metrics': config.List()
    })
})

metrics = {
    'vp': lambda trains, tau: stm.victor_purpura_dist(
        trains, 2.0 / tau, sort=False),
    'vr': lambda trains, tau: stm.van_rossum_dist(trains, tau, sort=False)
}

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
    results = sp.empty((
        len(cfg['metrics']), cfg['time_scales'].size, cfg['evaluation_points'],
        cfg['evaluation_points']))
    for m, metric in enumerate(cfg['metrics']):
        for t, tau in enumerate(cfg['time_scales']):
            logger.info("  for time scale %s" % str(tau))
            dist_mat = metrics[metric](
                list(itertools.chain(*trains_by_rate)), tau)
            for i in xrange(cfg['evaluation_points']):
                i_start = cfg['spike_trains_per_rate'] * i
                i_stop = i_start + cfg['spike_trains_per_rate']
                for j in xrange(i, cfg['evaluation_points']):
                    j_start = cfg['spike_trains_per_rate'] * j
                    j_stop = j_start + cfg['spike_trains_per_rate']
                    results[m, t, i, j] = results[m, t, j, i] = sp.mean(
                        dist_mat[i_start:i_stop, j_start:j_stop])
    return results


def plot(results):
    plt.figure()
    for m, metric in enumerate(cfg['metrics']):
        for t, tau in enumerate(cfg['time_scales']):
            plot_idx = len(cfg['time_scales']) * m + t + 1
            plt.subplot(len(cfg['metrics']), len(cfg['time_scales']), plot_idx)
            if m == 0:
                plt.title(r"$\tau = %s$" % str(tau))
            if t == 0:
                plt.ylabel(metric)
            plt.imshow(
                results[m, t], origin='lower', cmap=cm.get_cmap('hot'),
                extent=(0, cfg['max_rate'].magnitude,
                        0, cfg['max_rate'].magnitude))
            plt.colorbar()

logger.info("Section 3.1")
plot(calc_metric(gen_trains()))
plt.show()
