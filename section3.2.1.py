
#!/usr/bin/env python

from __future__ import division
from joblib import Memory, Parallel, delayed
from metrics import metrics
import argparse
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
import sys

name = 'section3.2.1'
memory = Memory(cachedir='cache', verbose=1)
logging.basicConfig()
logger = logging.getLogger(name)


def gen_trains(interval_length, rates, num_trials):
    logger.info("Generating spike trains")
    trains = []
    for i in xrange(num_trials):
        trains.append(stools.st_concatenate([stg.gen_homogeneous_poisson(
            rate, t_start=j * interval_length, t_stop=(j + 1) * interval_length)
            for j, rate in enumerate(rates)]))
    return trains


@memory.cache
def calc_single_metric(trains, metric, tau):
    logger.info("Calculating metric %s for time_scale %s." % (metric, str(tau)))
    return metrics[metric](trains, tau)


@memory.cache
def calc_probability_matrix(trains_a, trains_b, metric, tau, z):
    dist_mat = calc_single_metric(
        list(itertools.chain(trains_a, trains_b)), metric, tau) ** z
    classification = sp.argmin(sp.vstack((
        sp.mean(dist_mat[:len(trains_a), :], axis=0),
        sp.mean(dist_mat[len(trains_a):, :], axis=0))) ** (1 / z), axis=0)
    confusion = sp.empty((2, 2))
    confusion[0, 0] = sp.sum(classification[:len(trains_a)] == 0)
    confusion[1, 0] = sp.sum(classification[:len(trains_a)] == 1)
    confusion[0, 1] = sp.sum(classification[len(trains_a):] == 0)
    confusion[1, 1] = sp.sum(classification[len(trains_a):] == 1)
    assert len(trains_a) == len(trains_b)
    return confusion / 2.0 / len(trains_a)


def calc_mutual_information(probability_mat):
    marginals = sp.outer(
            sp.sum(probability_mat, axis=1),
            sp.sum(probability_mat, axis=0))
    p = probability_mat[probability_mat != 0.0]
    m = marginals[probability_mat != 0.0]
    return sp.sum(p * sp.log(p / m))


def calc_stimuli_entropy():
    return -2.0 * 0.5 * sp.log(0.5)

def calc_uncertainty_reduction(mutual_information):
    return mutual_information / calc_stimuli_entropy()


def run_experiment(cfg, experiment_idx):
    exp_cfg = cfg['experiments'][experiment_idx]
    trains_a = [gen_trains(
        cfg['interval_length'], exp_cfg['rates_a'], cfg['num_trials'])
        for i in xrange(cfg['repetitions'])]
    trains_b = [gen_trains(
        cfg['interval_length'], exp_cfg['rates_b'], cfg['num_trials'])
        for i in xrange(cfg['repetitions'])]

    results = {}
    param_sets = itertools.product(cfg['metrics'], cfg['zs'])
    for metric, z in param_sets:
        results[(metric, z)] = sp.array([[calc_uncertainty_reduction(
            calc_mutual_information(calc_probability_matrix(
                trains_a[i], trains_b[i], metric, tau, z)))
            for i in xrange(cfg['repetitions'])]
            for tau in cfg['time_scales']])
    return results


def run_experiments(cfg):
    return [run_experiment(cfg, i) for i in xrange(len(cfg['experiments']))]


def plot_stparams(interval_length, rates):
    for i, rate in enumerate(rates):
        plt.plot(
            [i * interval_length, (i + 1) * interval_length],
            [rate, rate])


def plot(cfg, results):
    for e, experiment in enumerate(cfg['experiments']):
        plt.subplot(
            len(cfg['metrics']) + 1, len(cfg['experiments']),
            e * len(cfg['metrics']) + 1)
        plot_stparams(cfg['interval_length'], experiment['rates_a'])
        plot_stparams(cfg['interval_length'], experiment['rates_b'])

        for m, metric in enumerate(cfg['metrics']):
            plt.subplot(
                len(cfg['metrics']) + 1, len(cfg['experiments']),
                e * len(cfg['metrics']) + m + 2)
            plt.semilogx()
            for z in cfg['zs']:
                plt.errorbar(
                    cfg['time_scales'], sp.mean(results[e][(metric, z)], axis=1),
                    yerr=sp.std(results[e][(metric, z)], axis=1))


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    logger.info("Section 3.1")

    config_spec = config.ConfigSpec({
        name: config.ConfigSpec({
            'num_trials': config.Integer(),
            'repetitions': config.Integer(),
            'interval_length': config.Quantity(pq.ms),
            'time_scales': config.Quantity(pq.ms),
            'metrics': config.List(),
            'zs': config.List(),
            'experiments': config.ConfigList({
                'name': config.String(),
                'rates_a': config.Quantity(pq.Hz),
                'rates_b': config.Quantity(pq.Hz)
            })
        })
    })

    parser = argparse.ArgumentParser(
        description="Reproduce %s of Chicharro 2011" % name)
    parser.add_argument(
        'conffile', type=str, nargs=1, help="Path to configuration file.")
    parser.add_argument(
        '-j', '--jobs', nargs=1, default=[1], type=int,
        help="Number of processes for parallelization.")
    parser.add_argument(
        '-o', '--output', nargs=1, type=str, help="Output file for plot.")
    parser.add_argument(
        '-s', '--show', action='store_true',
        help="Show the plot after creation")
    args = parser.parse_args()

    if args.output is None and not args.show:
        print "Error: You should provide either the -o or -s option. Use -h " +\
            "for help"
        sys.exit(-1)

    with open(args.conffile[0]) as config_file:
        cfg = config.load(config_spec, config_file)[name]

    plot(cfg, run_experiments(cfg))
    if args.output is not None:
        plt.savefig(args.output[0])
    if args.show:
        plt.show()
