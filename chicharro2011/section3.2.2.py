#!/usr/bin/env python

from __future__ import division
from joblib import Memory, Parallel, delayed
from matplotlib.ticker import FuncFormatter
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

name = 'section3.2.2'
memory = Memory(cachedir='cache', verbose=0)
logging.basicConfig()
logger = logging.getLogger(name)


def gen_single_trial(interval_lengths, rates):
    """ Generates a single spike train with intervals of length
    `interval_lengths` and the firing rates given in `rates`.
    """
    boundaries = sp.ones(len(interval_lengths) + 1) * pq.s
    boundaries[1:] = [l.rescale(boundaries.units) for l in interval_lengths]
    rates = rates[sp.nonzero(boundaries[1:])]
    boundaries = boundaries[sp.nonzero(boundaries)]
    boundaries[0] = 0.0 * pq.s
    boundaries = sp.cumsum(boundaries)
    return stools.st_concatenate([stg.gen_homogeneous_poisson(
        rate, t_start=boundaries[i], t_stop=boundaries[i + 1])
        for i, rate in enumerate(rates)])


def gen_trains(interval_lengths, rates, num_trials, num_repetitions):
    """ Generates a list of `num_repititions` lists of `num_trials` spike
    trains. Each spike train will have intervals with length `interval_lengths`
    and the firing rates given in `rates`.
    """
    return [[gen_single_trial(interval_lengths, rates)
            for j in xrange(num_trials)] for i in xrange(num_repetitions)]


@memory.cache
def gen_trains_pair(
        interval_lengths, rates_a, rates_b, num_trials, num_repetitions):
    """ Calls :func:`gen_trains` twice with `rates` set to `rates_a` and
    `rates_b` once. All other parameters passed to this function directly
    correspond to :func:`gen_trains`.
    """
    logger.info("Generating spike trains")
    return gen_trains(interval_lengths, rates_a, num_trials, num_repetitions), \
        gen_trains(interval_lengths, rates_b, num_trials, num_repetitions)


@memory.cache
def calc_single_metric(trains, metric, tau):
    """ Calculates a metric on spike trains.

    :param list trains: Spike trains to calculate the metric on.
    :param str metric: The metric to calculate. Has to be a key in
        :const:`metrics.metrics`.
    :param tau: Time scale parameter for the metric.
    :type tau: Quantity scalar.
    :returns: The metric for all pairs of spike trains.
    :rtype: 2-D array
    """
    logger.info("Calculating metric %s for time_scale %s." % (metric, str(tau)))
    return metrics[metric](trains, tau)


@memory.cache
def calc_probability_matrix(trains_a, trains_b, metric, tau, z):
    """ Calculates the probability matrix that one spike train from stimulus X
    will be classified as spike train from stimulus Y.

    :param list trains_a: Spike trains of stimulus A.
    :param list trains_b: Spike trains of stimulus B.
    :param str metric: Metric to base the classification on. Has to be a key in
        :const:`metrics.metrics`.
    :param tau: Time scale parameter for the metric.
    :type tau: Quantity scalar.
    :param float z: Exponent parameter for the classifier.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero")
        dist_mat = calc_single_metric(trains_a + trains_b, metric, tau) ** z
    dist_mat[sp.diag_indices_from(dist_mat)] = 0.0

    assert len(trains_a) == len(trains_b)
    l = len(trains_a)
    classification_of_a = sp.argmin(sp.vstack((
        sp.sum(dist_mat[:l, :l], axis=0) / (l - 1),
        sp.sum(dist_mat[l:, :l], axis=0) / l)) ** (1.0 / z), axis=0)
    classification_of_b = sp.argmin(sp.vstack((
        sp.sum(dist_mat[:l, l:], axis=0) / l,
        sp.sum(dist_mat[l:, l:], axis=0) / (l - 1))) ** (1.0 / z), axis=0)
    confusion = sp.empty((2, 2))
    confusion[0, 0] = sp.sum(classification_of_a == 0)
    confusion[1, 0] = sp.sum(classification_of_a == 1)
    confusion[0, 1] = sp.sum(classification_of_b == 0)
    confusion[1, 1] = sp.sum(classification_of_b == 1)
    return confusion / 2.0 / l


def calc_mutual_information(probability_mat):
    """ Calculates the mutual information of stimulus and predicted stimulus
    from a probability matrix.

    :rtype: float
    """

    marginals = sp.outer(
        sp.sum(probability_mat, axis=1), sp.sum(probability_mat, axis=0))
    p = probability_mat[probability_mat != 0.0]
    m = marginals[probability_mat != 0.0]
    return sp.sum(p * sp.log(p / m))


def calc_stimuli_entropy():
    """ :returns: Stimulus entropy for setting with 2 different stimuli of equal
        probability.
    """
    return -2.0 * 0.5 * sp.log(0.5)


def calc_uncertainty_reduction(trains_a, trains_b, metric, z, tau):
    """ Returns the percentage of uncertainty reduction by the stimulus
    prediction.

    :param list trains_a: Spike trains of stimulus A.
    :param list trains_b: Spike trains of stimulus B.
    :param str metric: Metric to base the classification of stimuli on for the
        calculation. Has to be a key in :const:`metrics.metrics`.
    :param z: Exponent parameter for the classifier.
    :param tau: Time scale parameter for the metric.
    :type tau: Quantity scalar.
    :returns: The percentage of uncertainty reduction.
    :rtype: float
    """

    return calc_mutual_information(calc_probability_matrix(
        trains_a, trains_b, metric, tau, z)) / calc_stimuli_entropy()


def run_exp_for_interval_lengths(interval_lengths, rates_a, rates_b, cfg):
    logger.info("Interval lengths %s" % str(interval_lengths))

    trains_a, trains_b = gen_trains_pair(
        interval_lengths, rates_a, rates_b, cfg['num_trials'],
        cfg['repetitions'])

    param_sets = itertools.product(
        cfg['time_scales'], xrange(cfg['repetitions']))
    result_list = [calc_uncertainty_reduction(
        trains_a[r], trains_b[r], cfg['metric'], cfg['z'], tau)
        for tau, r in param_sets]

    param_sets = xrange(cfg['repetitions'])
    result_list_inftau = [calc_uncertainty_reduction(
        trains_a[r], trains_b[r], cfg['metric'], cfg['z'], sp.inf * pq.s)
        for r in param_sets]

    return \
        sp.reshape(result_list, (len(cfg['time_scales']), cfg['repetitions'])), \
        sp.reshape(result_list_inftau, (cfg['repetitions'],))


def run_single_experiment(cfg, experiment_idx, n_jobs=1):
    exp_cfg = cfg['experiments'][experiment_idx]

    param_sets = itertools.product(cfg['l2_lengths'], cfg['l4_lengths'])
    result_list, result_list_inftau = zip(
        *Parallel(n_jobs)(delayed(run_exp_for_interval_lengths)(
            [cfg['l1_length'], l2, cfg['l3_length'], l4],
            exp_cfg['rates_a'], exp_cfg['rates_b'], cfg)
            for l2, l4 in param_sets))

    return \
        sp.reshape(
            result_list,
            (len(cfg['l2_lengths']), len(cfg['l4_lengths']),
             len(cfg['time_scales']), cfg['repetitions'])), \
        sp.reshape(
            result_list_inftau,
            (len(cfg['l2_lengths']), len(cfg['l4_lengths']),
             cfg['repetitions']))


def run_experiments(cfg, n_jobs=1):
    return zip(
        *[run_single_experiment(cfg, i, n_jobs)
          for i in xrange(len(cfg['experiments']))])


def plot_stparams(interval_lengths, rates, color):
    """ Plot a spike train consisting of intervals of the length
    `interval_lengths` with firing rates given in `rates` in the color `color`.
    """

    for i, rate in enumerate(rates):
        plt.plot(
            [sp.sum(interval_lengths[:i]), sp.sum(interval_lengths[:i + 1])],
            [rate, rate], c=color)


def plot_uncertainty_reduction(cfg, results, results_inftau):
    z_colors = ['k', 'm', 'r', 'b', 'c', 'g']
    for e, experiment in enumerate(cfg['experiments']):
        for i, l4 in enumerate(cfg['l4_lengths']):
            plt.subplot(
                len(cfg['l4_lengths']) + 1, len(cfg['experiments']),
                i * len(cfg['experiments']) + e + 1)
            plt.ylim(0, 1)
            if e <= 0:
                plt.ylabel(str(l4))
            else:
                plt.yticks([])
            plt.semilogx()
            plt.xlim(min(cfg['time_scales']), 50 * max(cfg['time_scales']))
            for j, l2 in enumerate(cfg['l2_lengths']):
                plt.plot(
                    cfg['time_scales'], sp.mean(results[e][i, j], axis=1),
                    c=z_colors[j], marker='o', markersize=5)
                plt.plot(
                    10 * max(cfg['time_scales']),
                    sp.mean(results_inftau[e][i, j]), c=z_colors[i],
                    marker='o', markersize=5)

            if i >= len(cfg['l4_lengths']) - 1:
                plt.xlabel(r"$\tau / ms$")
                formatter = plt.gca().xaxis.get_major_formatter()

                def include_inf(x, pos):
                    if x > max(cfg['time_scales']).magnitude:
                        return 'inf'
                    else:
                        return formatter(x, pos)

                plt.gca().xaxis.set_major_formatter(FuncFormatter(include_inf))
            else:
                plt.xticks([])


def plot_param_per_l4_and_l2(values, err=None, c=None):
    color = 'r'
    for i, l4 in enumerate(cfg['l4_lengths']):
        color = 'b' if color == 'r' else 'r'
        for j, l2 in enumerate(cfg['l2_lengths']):
            x = i * (len(cfg['l2_lengths']) + 2) + j + 0.5
            if err is None:
                plt.plot(x, values[i, j], c='g', marker='o', markersize=5)
            else:
                plt.errorbar(
                    x, values[i, j], yerr=err[i, j],
                    c=color if c is None else c, marker='o', markersize=5)


def plot_bool_indicator_per_l4_and_l2(values, c='g'):
    """ Plots markers on top of the y-axis sorted by metric and z-value on the
    x-axis.

    :param values: Boolean array with shape (metrics, z-values) indicating
        where to plot markers.
    :type values: 2-D array
    :param c: Color to plot in.
    """

    for i, l4 in enumerate(cfg['l4_lengths']):
        for j, l2 in enumerate(cfg['l2_lengths']):
            x = i * (len(cfg['l2_lengths']) + 2) + j + 0.5
            if values[i, j]:
                plt.plot(
                    x, 2 * cfg['time_scales'][-1], c=c, marker='o',
                    markersize=5)


def plot_optimal_uncertainty_reduction(results_for_exp, results_for_exp_inftau):
    """ Plot the percentage of uncertainty reduction of the optimal classifiers.

    :param results_for_exp: The results of one experiment as 4-D array of the
        shape (metrics, z-values, tau-values, experimental repetitions).
    :type results_for_exp: 4-D array
    :param result_list_inftau: The results of one experiment for `tau = inf` as
        3-D array of the shape (metrics, z-values, experimental repetitions).
    :type results_for_exp_inftau: 3-D array.
    """
    plt.ylim(0, 1)

    plot_param_per_l4_and_l2(
        sp.mean(sp.amax(results_for_exp, axis=2), axis=2),
        sp.std(sp.amax(results_for_exp, axis=2), axis=2))
    plot_param_per_l4_and_l2(sp.mean(results_for_exp_inftau, axis=2), c='g')


def plot_optimal_tau(results_for_exp, results_for_exp_inftau):
    """ Plot the mean of optimal taus.

    :param results_for_exp: The results of one experiment as 4-D array of the
        shape (metrics, z-values, tau-values, experimental repetitions).
    :type results_for_exp: 4-D array
    :param result_list_inftau: The results of one experiment for `tau = inf` as
        3-D array of the shape (metrics, z-values, experimental repetitions).
    :type results_for_exp_inftau: 3-D array.
    """
    values = sp.empty((results_for_exp.shape[0], results_for_exp.shape[1]))
    err = sp.empty((results_for_exp.shape[0], results_for_exp.shape[1], 2, 1))
    num_trials = results_for_exp.shape[3]
    for i, l4 in enumerate(cfg['l4_lengths']):
        for j, l2 in enumerate(cfg['l2_lengths']):
            r = [sp.mean(cfg['time_scales'][
                results_for_exp[i, j, :, t] == results_for_exp[i, j, :, t].max()])
                for t in xrange(num_trials)]
            values[i, j] = sp.mean(r)
            err[i, j, 0] = values[i, j] - min(r).magnitude
            err[i, j, 1] = max(r).magnitude - values[i, j]
    plot_param_per_l4_and_l2(values, err)
    plot_bool_indicator_per_l4_and_l2(
        sp.any(results_for_exp_inftau >= results_for_exp.max(axis=2), axis=2))


def plot_optimal_tau_for_mean_uncertainty_reduction(
        results_for_exp, results_for_exp_inftau):
    """ Plot the optimal tau for the mean of uncertainty reduction.

    :param results_for_exp: The results of one experiment as 4-D array of the
        shape (metrics, z-values, tau-values, experimental repetitions).
    :type results_for_exp: 4-D array
    :param result_list_inftau: The results of one experiment for `tau = inf` as
        3-D array of the shape (metrics, z-values, experimental repetitions).
    :type results_for_exp_inftau: 3-D array.
    """
    values = sp.empty((results_for_exp.shape[0], results_for_exp.shape[1]))
    err = sp.empty((results_for_exp.shape[0], results_for_exp.shape[1], 2, 1))
    mark = sp.empty((results_for_exp.shape[0], results_for_exp.shape[1]))
    for i, l4 in enumerate(cfg['l4_lengths']):
        for j, l2 in enumerate(cfg['l2_lengths']):
            r = sp.mean(results_for_exp[i, j], axis=1)
            mark[i, j] = r.max()
            values[i, j] = sp.mean(cfg['time_scales'][r == r.max()]).magnitude
            r = cfg['time_scales'][r > 0.8 * r.max()]
            err[i, j, 0] = values[i, j] - min(r).magnitude
            err[i, j, 1] = max(r).magnitude + values[i, j]
    plot_param_per_l4_and_l2(values, err)
    plot_bool_indicator_per_l4_and_l2(
        sp.mean(results_for_exp_inftau, axis=2) >= mark)


def plot_optima(cfg, results, results_inftau):
    """ Plot optima for uncertainty reduction and tau values.

    :param dict cfg: Configuration.
    :param list results: List of results with one element per experiment. Each
        element has to be a 4-D array of the shape (metrics, z-values,
        tau-values, experimental repetitions).
    :param list result_list_inftau: List of results for `tau = inf` with one
        element per experiment. Each element has to be a 3-D array of the shape
        (metrics, z-values, experimental repetitions).
    """

    def subplot_args(experiment_idx, row):
        return 4, len(cfg['experiments']), \
            row * len(cfg['experiments']) + experiment_idx + 1

    for e, experiment in enumerate(cfg['experiments']):
            def set_rowlabels(rowname):
                if e <= 0:
                    plt.ylabel(rowname)
                else:
                    plt.yticks([])

            plt.subplot(*subplot_args(e, 0))

            plt.title(experiment['name'])
            plt.xlabel("time / ms")
            plt.ylim(0, 210)
            if e > 0:
                plt.yticks([])
            plot_stparams(
                [cfg['l1_length'], cfg['l2_lengths'][1], cfg['l3_length'],
                 cfg['l4_lengths'][1]], experiment['rates_a'], 'm')
            plot_stparams(
                [cfg['l1_length'], cfg['l2_lengths'][1], cfg['l3_length'],
                 cfg['l4_lengths'][1]], experiment['rates_b'], 'k')

            plt.subplot(*subplot_args(e, 1))
            set_rowlabels(r"$\langle I^* \rangle$")
            plt.xticks([])
            plt.xlim(
                0,
                (len(cfg['l4_lengths']) + 0.5) * (len(cfg['l2_lengths']) + 2))
            plot_optimal_uncertainty_reduction(results[e], results_inftau[e])

            plt.subplot(*subplot_args(e, 2))
            plt.ylim([10 * cfg['time_scales'][0], 3 * cfg['time_scales'][-1]])
            plt.semilogy()
            set_rowlabels(r"$\langle \tau^* \rangle$")
            plt.xticks([])
            plt.xlim(
                0,
                (len(cfg['l4_lengths']) + 0.5) * (len(cfg['l2_lengths']) + 2))
            plot_optimal_tau(results[e], results_inftau[e])

            plt.subplot(*subplot_args(e, 3))
            plt.ylim([10 * cfg['time_scales'][0], 3 * cfg['time_scales'][-1]])
            plt.semilogy()
            set_rowlabels(r"$\tau^*_{\langle I \rangle}$")
            plt.xlim(
                0,
                (len(cfg['l4_lengths']) + 0.5) * (len(cfg['l2_lengths']) + 2))
            plot_optimal_tau_for_mean_uncertainty_reduction(
                results[e], results_inftau[e])

            plt.xticks(
                (sp.arange(len(cfg['l4_lengths'])) + 0.5) *
                (len(cfg['l2_lengths']) + 2),
                ['%s' % l for l in cfg['l4_lengths']])


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    logger.info("Section 3.2.2")

    config_spec = config.ConfigSpec({
        name: config.ConfigSpec({
            'num_trials': int,
            'repetitions': int,
            'l1_length': config.Quantity(pq.ms),
            'l2_lengths': config.Quantity(pq.ms),
            'l3_length': config.Quantity(pq.ms),
            'l4_lengths': config.Quantity(pq.ms),
            'time_scales': config.QuantityLogRange(pq.ms),
            'metric': str,
            'z': float,
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
        '-o', '--output', nargs=2, type=str, help="Output file for plot.")
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

    results = run_experiments(cfg, args.jobs[0])

    plt.figure()
    plot_uncertainty_reduction(cfg, *results)
    if args.output is not None:
        plt.savefig(args.output[0])

    plt.figure()
    plot_optima(cfg, *results)
    if args.output is not None:
        plt.savefig(args.output[1])

    if args.show:
        plt.show()
