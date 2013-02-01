#!/usr/bin/env python

from __future__ import division
from joblib import Memory, Parallel, delayed
from metrics import metrics
import argparse
import config
import itertools
import logging
import matplotlib.pyplot as plt
import quantities as pq
import scipy as sp
import spykeutils.tools as stools
import spykeutils.spike_train_generation as stg
import sys
import warnings

name = 'section3.2.1'
memory = Memory(cachedir='cache', verbose=0)
logging.basicConfig()
logger = logging.getLogger(name)


def gen_single_trial(interval_length, rates):
    return stools.st_concatenate([stg.gen_homogeneous_poisson(
        rate, t_start=i * interval_length,
        t_stop=(i + 1) * interval_length) for i, rate in enumerate(rates)])


def gen_trains(interval_length, rates, num_trials, num_repetitions):
    return [[gen_single_trial(interval_length, rates)
            for j in xrange(num_trials)] for i in xrange(num_repetitions)]


@memory.cache
def gen_trains_pair(
        interval_length, rates_a, rates_b, num_trials, num_repetitions):
    logger.info("Generating spike trains")
    return gen_trains(interval_length, rates_a, num_trials, num_repetitions), \
        gen_trains(interval_length, rates_b, num_trials, num_repetitions)


@memory.cache
def calc_single_metric(trains, metric, tau):
    logger.info("Calculating metric %s for time_scale %s." % (metric, str(tau)))
    return metrics[metric](trains, tau)


@memory.cache
def calc_probability_matrix(trains_a, trains_b, metric, tau, z):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero")
        dist_mat = calc_single_metric(
            list(itertools.chain(trains_a, trains_b)), metric, tau) ** z
    dist_mat[sp.diag_indices(dist_mat.shape[0])] = 0.0
    classification = sp.argmin(sp.vstack((
        sp.mean(dist_mat[:len(trains_a), :], axis=0),
        sp.mean(dist_mat[len(trains_a):, :], axis=0))) ** (1.0 / z), axis=0)
    confusion = sp.empty((2, 2))
    confusion[0, 0] = sp.sum(classification[:len(trains_a)] == 0)
    confusion[1, 0] = sp.sum(classification[:len(trains_a)] == 1)
    confusion[0, 1] = sp.sum(classification[len(trains_a):] == 0)
    confusion[1, 1] = sp.sum(classification[len(trains_a):] == 1)
    assert len(trains_a) == len(trains_b)
    return confusion / 2.0 / len(trains_a)


def calc_mutual_information(probability_mat):
    marginals = sp.outer(
        sp.sum(probability_mat, axis=1), sp.sum(probability_mat, axis=0))
    p = probability_mat[probability_mat != 0.0]
    m = marginals[probability_mat != 0.0]
    return sp.sum(p * sp.log(p / m))


def calc_stimuli_entropy():
    return -2.0 * 0.5 * sp.log(0.5)


def calc_uncertainty_reduction(trains_a, trains_b, metric, z, tau):
    return calc_mutual_information(calc_probability_matrix(
        trains_a, trains_b, metric, tau, z)) / calc_stimuli_entropy()


def run_single_experiment(cfg, experiment_idx, n_jobs=1):
    exp_cfg = cfg['experiments'][experiment_idx]
    trains_a, trains_b = gen_trains_pair(
        cfg['interval_length'], exp_cfg['rates_a'], exp_cfg['rates_b'],
        cfg['num_trials'], cfg['repetitions'])

    param_sets = itertools.product(
        cfg['metrics'], cfg['zs'], cfg['time_scales'],
        xrange(cfg['repetitions']))
    result_list = Parallel(n_jobs)(delayed(calc_uncertainty_reduction)(
        trains_a[r], trains_b[r], m, z, tau) for m, z, tau, r in param_sets)
    return sp.reshape(
        result_list,
        (len(cfg['metrics']), len(cfg['zs']), len(cfg['time_scales']),
         cfg['repetitions']))


def run_experiments(cfg, n_jobs=1):
    return [run_single_experiment(cfg, i, n_jobs)
            for i in xrange(len(cfg['experiments']))]


def run_single_experiment_inftau(cfg, experiment_idx, n_jobs=1):
    exp_cfg = cfg['experiments'][experiment_idx]
    trains_a, trains_b = gen_trains_pair(
        cfg['interval_length'], exp_cfg['rates_a'], exp_cfg['rates_b'],
        cfg['num_trials'], cfg['repetitions'])

    param_sets = itertools.product(
        cfg['metrics'], cfg['zs'], xrange(cfg['repetitions']))
    result_list = Parallel(n_jobs)(delayed(calc_uncertainty_reduction)(
        trains_a[r], trains_b[r], m, z, sp.inf) for m, z, r in param_sets)
    return sp.reshape(
        result_list, (len(cfg['metrics']), len(cfg['zs']), cfg['repetitions']))


def run_experiments_inftau(cfg, n_jobs=1):
    return [run_single_experiment_inftau(cfg, i, n_jobs)
            for i in xrange(len(cfg['experiments']))]


def plot_stparams(interval_length, rates, color):
    for i, rate in enumerate(rates):
        plt.plot(
            [i * interval_length, (i + 1) * interval_length],
            [rate, rate], c=color)


def plot(cfg, results):
    z_colors = ['k', 'r', 'b', 'g']
    for e, experiment in enumerate(cfg['experiments']):
        plt.subplot(len(cfg['metrics']) + 1, len(cfg['experiments']), e + 1)
        plt.title(experiment['name'])
        plt.xlabel("time / ms")
        plt.ylim(0, 110)
        plot_stparams(cfg['interval_length'], experiment['rates_a'], 'm')
        plot_stparams(cfg['interval_length'], experiment['rates_b'], 'k')

        for m, metric in enumerate(cfg['metrics']):
            plt.subplot(
                len(cfg['metrics']) + 1, len(cfg['experiments']),
                (m + 1) * len(cfg['experiments']) + e + 1)
            plt.ylabel("$%s$" % metric)
            plt.ylim(0, 1)
            if m >= len(cfg['metrics']) - 1:
                plt.xlabel(r"$\tau / ms$")
            plt.semilogx()
            for z in xrange(len(cfg['zs'])):
                if cfg['zs'][z] == -2:
                    plt.errorbar(
                        cfg['time_scales'], sp.mean(results[e][m, z], axis=1),
                        yerr=sp.std(results[e][m, z], axis=1), c=z_colors[z],
                        marker='o')
                else:
                    plt.plot(
                        cfg['time_scales'], sp.mean(results[e][m, z], axis=1),
                        c=z_colors[z], marker='o')


def plot2(cfg, results, results_inftau):
    for e, experiment in enumerate(cfg['experiments']):
            plt.subplot(
                3, len(cfg['experiments']), 0 * len(cfg['experiments']) + e + 1)
            plt.title(experiment['name'])

            if e <= 0:
                plt.ylabel(r"$\langle I^* \rangle$")
            plt.ylim(0, 1)

            color = 'r'
            for m, metric in enumerate(cfg['metrics']):
                color = 'b' if color == 'r' else 'r'
                for z in xrange(len(cfg['zs'])):
                    plt.errorbar(
                        m * len(cfg['zs']) + z + 0.5,
                        sp.mean(sp.amax(results[e][m, z], axis=0)),
                        yerr=sp.std(sp.amax(results[e][m, z], axis=0)),
                        c=color, marker='o')
                    plt.plot(
                        m * len(cfg['zs']) + z + 0.5,
                        sp.amax(results_inftau[e][m, z]), c='g', marker='o')

            plt.subplot(
                3, len(cfg['experiments']), 1 * len(cfg['experiments']) + e + 1)
            plt.semilogy()
            plt.ylim([cfg['time_scales'][0], 3 * cfg['time_scales'][-1]])
            if e <= 0:
                plt.ylabel(r"$\langle \tau^* \rangle$")
            color = 'r'
            for m, metric in enumerate(cfg['metrics']):
                color = 'b' if color == 'r' else 'r'
                for z in xrange(len(cfg['zs'])):
                    r = []
                    for t in xrange(results[e][m, z].shape[1]):
                        x = results[e][m, z, :, t]
                        r.append(sp.mean(cfg['time_scales'][x == x.max()]))
                    r2 = sp.mean(r)
                    plt.errorbar(
                        m * len(cfg['zs']) + z + 0.5,
                        r2, yerr=sp.vstack((r2-min(r).magnitude, max(r).magnitude-r2)),
                        c=color, marker='o')
                    if sp.any(results_inftau[e][m, z] > results[e][m, z]):
                        plt.plot(
                            m * len(cfg['zs']) + z + 0.5,
                            cfg['time_scales'][-1] * 2, c='g', marker='o')


            plt.subplot(
                3, len(cfg['experiments']), 2 * len(cfg['experiments']) + e + 1)
            plt.semilogy()
            plt.ylim([cfg['time_scales'][0], 3 * cfg['time_scales'][-1]])
            if e <= 0:
                plt.ylabel(r"$\tau^*_{\langle I \rangle}$")
            color = 'r'
            for m, metric in enumerate(cfg['metrics']):
                color = 'b' if color == 'r' else 'r'
                for z in xrange(len(cfg['zs'])):
                    r = sp.mean(results[e][m, z], axis=0)
                    x = sp.mean(cfg['time_scales'][r == r.max()])
                    r2 = cfg['time_scales'][r > 0.8 * r.max()]
                    plt.errorbar(
                        m * len(cfg['zs']) + z + 0.5,
                        x.magnitude, yerr=sp.vstack((x-min(r2), max(r2)-x)),
                        c=color, marker='o')
                    if sp.mean(results_inftau[e][m, z]) > r.max():
                        plt.plot(
                            m * len(cfg['zs']) + z + 0.5,
                            cfg['time_scales'][-1] * 2, c='g', marker='o')

            plt.xticks(
                (sp.arange(len(cfg['metrics'])) + 0.5) * len(cfg['zs']),
                cfg['metrics'])


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    logger.info("Section 3.2.1")

    config_spec = config.ConfigSpec({
        name: config.ConfigSpec({
            'num_trials': int,
            'repetitions': int,
            'interval_length': config.Quantity(pq.ms),
            'time_scales': config.QuantityLogRange(pq.ms),
            'metrics': list,
            'zs': list,
            'experiments': config.ConfigList({
                'name': str,
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

    plot2(cfg, run_experiments(
        cfg, args.jobs[0]), run_experiments_inftau(cfg, args.jobs[0]))
    if args.output is not None:
        plt.savefig(args.output[0])
    if args.show:
        plt.show()
