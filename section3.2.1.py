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

    param_sets = itertools.product(
        cfg['metrics'], cfg['zs'], xrange(cfg['repetitions']))
    result_list_inftau = Parallel(n_jobs)(delayed(calc_uncertainty_reduction)(
        trains_a[r], trains_b[r], m, z, sp.inf) for m, z, r in param_sets)

    return \
        sp.reshape(
            result_list,
            (len(cfg['metrics']), len(cfg['zs']), len(cfg['time_scales']),
             cfg['repetitions'])), \
        sp.reshape(
            result_list_inftau,
            (len(cfg['metrics']), len(cfg['zs']), cfg['repetitions']))


def run_experiments(cfg, n_jobs=1):
    return zip(
        *[run_single_experiment(cfg, i, n_jobs)
          for i in xrange(len(cfg['experiments']))])


def plot_stparams(interval_length, rates, color):
    for i, rate in enumerate(rates):
        plt.plot(
            [i * interval_length, (i + 1) * interval_length],
            [rate, rate], c=color)


def plot_uncertainty_reduction(cfg, results, results_inftau):
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


def plot_param_per_metric_and_z(values, err=None, c=None):
    color = 'r'
    for m, metric in enumerate(cfg['metrics']):
        color = 'b' if color == 'r' else 'r'
        for z in xrange(len(cfg['zs'])):
            x = m * len(cfg['zs']) + z + 0.5
            if err is None:
                plt.plot(x, values[m, z], c='g', marker='o')
            else:
                plt.errorbar(
                    x, values[m, z], yerr=err[m, z],
                    c=color if c is None else c, marker='o')


def plot_bool_indicator_per_metric_and_z(values, c='g'):
    for m, metric in enumerate(cfg['metrics']):
        for z in xrange(len(cfg['zs'])):
            x = m * len(cfg['zs']) + z + 0.5
            if values[m, z]:
                plt.plot(x, 2 * cfg['time_scales'][-1], c=c, marker='o')


def plot_optimal_uncertainty_reduction(results_for_exp, results_for_exp_inftau):
    plt.ylim(0, 1)

    plot_param_per_metric_and_z(
        sp.mean(sp.amax(results_for_exp, axis=2), axis=2),
        sp.std(sp.amax(results_for_exp, axis=2), axis=2))
    plot_param_per_metric_and_z(sp.amax(results_for_exp_inftau, axis=2), c='g')


def plot_optimal_tau(results_for_exp, results_for_exp_inftau):
    values = sp.empty((results_for_exp.shape[0], results_for_exp.shape[1]))
    err = sp.empty((results_for_exp.shape[0], results_for_exp.shape[1], 2, 1))
    num_trials = results_for_exp.shape[3]
    for m, metric in enumerate(cfg['metrics']):
        for z in xrange(len(cfg['zs'])):
            r = [sp.mean(cfg['time_scales'][
                results_for_exp[m, z, :, t] == results_for_exp[m, z, :, t].max()])
                for t in xrange(num_trials)]
            values[m, z] = sp.mean(r)
            err[m, z, 0] = values[m, z] - min(r).magnitude
            err[m, z, 1] = max(r).magnitude - values[m, z]
    plot_param_per_metric_and_z(values, err)
    plot_bool_indicator_per_metric_and_z(
        sp.any(results_for_exp_inftau > results_for_exp.max(axis=2), axis=2))


def plot_optimal_tau_for_mean_uncertainty_reduction(
        results_for_exp, results_for_exp_inftau):
    values = sp.empty((results_for_exp.shape[0], results_for_exp.shape[1]))
    err = sp.empty((results_for_exp.shape[0], results_for_exp.shape[1], 2, 1))
    mark = sp.empty((results_for_exp.shape[0], results_for_exp.shape[1]))
    for m, metric in enumerate(cfg['metrics']):
        for z in xrange(len(cfg['zs'])):
            r = sp.mean(results_for_exp[m, z], axis=0)
            mark[m, z] = r.max()
            values[m, z] = sp.mean(cfg['time_scales'][r == r.max()]).magnitude
            r = cfg['time_scales'][r > 0.8 * r.max()]
            err[m, z, 0] = values[m, z] - min(r).magnitude
            err[m, z, 1] = max(r).magnitude + values[m, z]
    plot_param_per_metric_and_z(values, err)
    plot_bool_indicator_per_metric_and_z(
        sp.mean(results_for_exp_inftau, axis=2) > mark)


def plot_optima(cfg, results, results_inftau):
    for e, experiment in enumerate(cfg['experiments']):
            plt.subplot(
                3, len(cfg['experiments']), 0 * len(cfg['experiments']) + e + 1)
            plt.title(experiment['name'])

            if e <= 0:
                plt.ylabel(r"$\langle I^* \rangle$")

            plot_optimal_uncertainty_reduction(results[e], results_inftau[e])

            plt.subplot(
                3, len(cfg['experiments']), 1 * len(cfg['experiments']) + e + 1)
            plt.semilogy()
            plt.ylim([cfg['time_scales'][0], 3 * cfg['time_scales'][-1]])
            if e <= 0:
                plt.ylabel(r"$\langle \tau^* \rangle$")
            plot_optimal_tau(results[e], results_inftau[e])

            plt.subplot(
                3, len(cfg['experiments']), 2 * len(cfg['experiments']) + e + 1)
            plt.semilogy()
            plt.ylim([cfg['time_scales'][0], 3 * cfg['time_scales'][-1]])
            if e <= 0:
                plt.ylabel(r"$\tau^*_{\langle I \rangle}$")
            plot_optimal_tau_for_mean_uncertainty_reduction(
                results[e], results_inftau[e])

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
        '-o', '--output', nargs=2, type=str, help="Output files for plots.")
    parser.add_argument(
        '-s', '--show', action='store_true',
        help="Show the plot_uncertainty_reduction after creation")
    args = parser.parse_args()

    if args.output is None and not args.show:
        print "Error: You should provide either the -o or -s option. Use -h " +\
            "for help"
        sys.exit(-1)

    with open(args.conffile[0]) as config_file:
        cfg = config.load(config_spec, config_file)[name]

    results = run_experiments(cfg, args.jobs[0])

    plot_uncertainty_reduction(cfg, *results)
    if args.output is not None:
        plt.savefig(args.output[0])

    plot_optima(cfg, *results)
    if args.output is not None:
        plt.savefig(args.output[1])

    if args.show:
        plt.show()
