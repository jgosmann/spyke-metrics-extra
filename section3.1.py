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
import spykeutils.tools as stools
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


def binning_distance(trains, tau):
    sampling_rate = 1.0 / tau
    binned, dummy = stools.bin_spike_trains({0: trains}, sampling_rate)
    binned = sp.tile(sp.vstack(trains), (len(trains), 1, 1))
    return sp.sum((binned - sp.rollaxis(binned, 1)) ** 2, axis=2)


def normalized_vp_dist(trains, tau):
    num_spikes = sp.atleast_2d(sp.asarray([st.size for st in trains]))
    return stm.victor_purpura_dist(
        trains, 2.0 / tau, sort=False) / (num_spikes + num_spikes.T)


metrics = {
    'D_V': lambda trains, tau: stm.victor_purpura_dist(
        trains, 2.0 / tau, sort=False),
    'D_V*': normalized_vp_dist,
    'D_R': lambda trains, tau: stm.van_rossum_dist(trains, tau, sort=False),
    'D_B': binning_distance,
    'D_ES': lambda trains, tau: stm.event_synchronization(
        trains, tau, sort=False),
    'D_HM': stm.hunter_milton_similarity
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


# TODO split this function, so that we have one call per metric and tau, this
# allows to reuse more cached results.
@memory.cache
def calc_single_metric(trains_by_rate, metric, tau):
    result = sp.empty((cfg['evaluation_points'], cfg['evaluation_points']))
    dist_mat = metric(list(itertools.chain(*trains_by_rate)), tau)
    for i in xrange(cfg['evaluation_points']):
        i_start = cfg['spike_trains_per_rate'] * i
        i_stop = i_start + cfg['spike_trains_per_rate']
        for j in xrange(i, cfg['evaluation_points']):
            j_start = cfg['spike_trains_per_rate'] * j
            j_stop = j_start + cfg['spike_trains_per_rate']
            result[i, j] = result[j, i] = sp.mean(
                dist_mat[i_start:i_stop, j_start:j_stop])
    return result


def calc_metrics(trains_by_rate):
    logger.info("Calculating metrics")
    results = sp.empty((
        len(cfg['metrics']), cfg['time_scales'].size, cfg['evaluation_points'],
        cfg['evaluation_points']))
    for m, metric in enumerate(cfg['metrics']):
        logger.info("  metric %s" % metric)
        for t, tau in enumerate(cfg['time_scales']):
            logger.info("    for time scale %s" % str(tau))
            results[m, t, :, :] = calc_single_metric(
                trains_by_rate, metrics[metric], tau)
    return results


def plot(results):
    logger.info("Plotting")
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
plot(calc_metrics(gen_trains()))
plt.show()
