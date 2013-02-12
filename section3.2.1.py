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

name = 'section3.2.1'
memory = Memory(cachedir='cache', verbose=0)
logging.basicConfig()
logger = logging.getLogger(name)


def gen_single_trial(interval_length, rates):
    """ Generates a single spike train with intervals of length
    `interval_length` and the firing rates given in `rates`.
    """
    return stools.st_concatenate([stg.gen_homogeneous_poisson(
        rate, t_start=i * interval_length,
        t_stop=(i + 1) * interval_length) for i, rate in enumerate(rates)])


def gen_trains(interval_length, rates, num_trials, num_repetitions):
    """ Generates a list of `num_repititions` lists of `num_trials` spike
    trains. Each spike train will have intervals with length `interval_length`
    and the firing rates given in `rates`.
    """
    return [[gen_single_trial(interval_length, rates)
            for j in xrange(num_trials)] for i in xrange(num_repetitions)]


@memory.cache
def gen_trains_pair(
        interval_length, rates_a, rates_b, num_trials, num_repetitions):
    """ Calls :func:`gen_trains` twice with `rates` set to `rates_a` and
    `rates_b` once. All other parameters passed to this function directly
    correspond to :func:`gen_trains`.
    """
    logger.info("Generating spike trains")
    return gen_trains(interval_length, rates_a, num_trials, num_repetitions), \
        gen_trains(interval_length, rates_b, num_trials, num_repetitions)


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


def run_single_experiment(cfg, experiment_idx, n_jobs=1):
    """ Run `experiment_idx`-th experiment defined in the configuration `cfg`
    with `n_jobs` parallel processes.

    :returns: Tuple with:
        * The percentages of uncertainty reduction as an array of the shape
          (metrics, z-values, tau-values, repetitions of the experiment)
        * The percentages of uncertainty reduction for `tau = inf` as an array
          of the shape (metrics, z-values, repetitions of the experiment)
    :rtype: (4-D array, 3-D array)
    """

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
        trains_a[r], trains_b[r], m, z, sp.inf * pq.s) for m, z, r in param_sets)

    return \
        sp.reshape(
            result_list,
            (len(cfg['metrics']), len(cfg['zs']), len(cfg['time_scales']),
             cfg['repetitions'])), \
        sp.reshape(
            result_list_inftau,
            (len(cfg['metrics']), len(cfg['zs']), cfg['repetitions']))


def run_experiments(cfg, n_jobs=1):
    """ Run all experiments defined in the configuration `cfg` with `n_jobs`
    parallel processes.

    :returns: Tuple with:
        * List of the percentages of uncertainty reduction as an array of the
          shape (metrics, z-values, tau-values, repetitions of the experiment)
        * List of the percentages of uncertainty reduction for `tau = inf` as
          an array of the shape (metrics, z-values, repetitions of the
          experiment)
    :rtype: (list of 4-D arrays, list of 3-D arrays)
    """

    return zip(
        *[run_single_experiment(cfg, i, n_jobs)
          for i in xrange(len(cfg['experiments']))])


def plot_stparams(interval_length, rates, color):
    """ Plot a spike train consisting of intervals of the length
    `interval_length` with firing rates given in `rates` in the color `color`.
    """

    for i, rate in enumerate(rates):
        plt.plot(
            [i * interval_length, (i + 1) * interval_length],
            [rate, rate], c=color)


def plot_uncertainty_reduction(cfg, results, results_inftau):
    """ Plot the percentage of reduction of uncertainty.

    :param dict cfg: Configuration.
    :param list results: List of results with one element per experiment. Each
        element has to be a 4-D array of the shape (metrics, z-values,
        tau-values, experimental repetitions).
    :param list result_list_inftau: List of results for `tau = inf` with one
        element per experiment. Each element has to be a 3-D array of the shape
        (metrics, z-values, experimental repetitions).
    """

    z_colors = ['k', 'm', 'r', 'b', 'c', 'g']
    for e, experiment in enumerate(cfg['experiments']):
        plt.subplot(len(cfg['plot_uncertainty_reduction']) + 1, len(cfg['experiments']), e + 1)
        plt.title(experiment['name'])
        plt.xlabel("time / ms")
        plt.ylim(0, 110)
        if e > 0:
            plt.yticks([])
        plot_stparams(cfg['interval_length'], experiment['rates_a'], 'm')
        plot_stparams(cfg['interval_length'], experiment['rates_b'], 'k')

        for m, metric in enumerate(cfg['plot_uncertainty_reduction']):
            plt.subplot(
                len(cfg['plot_uncertainty_reduction']) + 1, len(cfg['experiments']),
                (m + 1) * len(cfg['experiments']) + e + 1)
            plt.ylim(0, 1)
            if e <= 0:
                plt.ylabel("$%s$" % metric)
            else:
                plt.yticks([])
            plt.semilogx()
            plt.xlim(min(cfg['time_scales']), 50 * max(cfg['time_scales']))
            for z in xrange(len(cfg['zs'])):
                if not cfg['zs'][z] in cfg['plot_z_curves']:
                    continue
                if cfg['zs'][z] == -2:
                    plt.errorbar(
                        cfg['time_scales'], sp.mean(results[e][m, z], axis=1),
                        yerr=sp.std(results[e][m, z], axis=1), c=z_colors[z],
                        marker='o', markersize=5)
                    plt.errorbar(
                        10 * max(cfg['time_scales']),
                        sp.mean(results_inftau[e][m, z]),
                        yerr=sp.std(results_inftau[e][m, z]), c=z_colors[z],
                        marker='o', markersize=5)
                else:
                    plt.plot(
                        cfg['time_scales'], sp.mean(results[e][m, z], axis=1),
                        c=z_colors[z], marker='o', markersize=5)
                    plt.plot(
                        10 * max(cfg['time_scales']),
                        sp.mean(results_inftau[e][m, z]), c=z_colors[z],
                        marker='o', markersize=5)

            if m >= len(cfg['plot_uncertainty_reduction']) - 1:
                plt.xlabel(r"$\tau / ms$")
                formatter = plt.gca().xaxis.get_major_formatter()

                def include_inf(x, pos):
                    if x > max(cfg['time_scales']).magnitude:
                        return 'inf'
                    else:
                        return formatter(x, pos)

                plt.gca().xaxis.set_major_formatter(FuncFormatter(include_inf))
                #plt.xticks(locs[:-1], labels[:-1])
            else:
                plt.xticks([])


def plot_param_per_metric_and_z(values, err=None, c=None):
    """ Plots values on the y-axis sorted by metric and z-value on the x-axis.

    :param values: Array with shape (metrics, z-values) with values to plot.
    :type values: 2-D array
    :param err: Array with shape (metrics, z-values) or (metrics, z-values, 2)
        denoting the error bars to plot.
    :type err: 2-D or 3-D array
    :param c: Color to plot in. If `None`, blue and red will be used alternating
        for each metric.
    """

    color = 'r'
    for m, metric in enumerate(cfg['metrics']):
        color = 'b' if color == 'r' else 'r'
        for z in xrange(len(cfg['zs'])):
            x = m * (len(cfg['zs']) + 2) + z + 0.5
            if err is None:
                plt.plot(x, values[m, z], c='g', marker='o', markersize=5)
            else:
                plt.errorbar(
                    x, values[m, z], yerr=err[m, z],
                    c=color if c is None else c, marker='o', markersize=5)


def plot_bool_indicator_per_metric_and_z(values, c='g'):
    """ Plots markers on top of the y-axis sorted by metric and z-value on the
    x-axis.

    :param values: Boolean array with shape (metrics, z-values) indicating
        where to plot markers.
    :type values: 2-D array
    :param c: Color to plot in.
    """

    for m, metric in enumerate(cfg['metrics']):
        for z in xrange(len(cfg['zs'])):
            x = m * (len(cfg['zs']) + 2) + z + 0.5
            if values[m, z]:
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

    plot_param_per_metric_and_z(
        sp.mean(sp.amax(results_for_exp, axis=2), axis=2),
        sp.std(sp.amax(results_for_exp, axis=2), axis=2))
    plot_param_per_metric_and_z(sp.mean(results_for_exp_inftau, axis=2), c='g')


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
    for m, metric in enumerate(cfg['metrics']):
        for z in xrange(len(cfg['zs'])):
            r = sp.mean(results_for_exp[m, z], axis=1)
            mark[m, z] = r.max()
            values[m, z] = sp.mean(cfg['time_scales'][r == r.max()]).magnitude
            r = cfg['time_scales'][r > 0.8 * r.max()]
            err[m, z, 0] = values[m, z] - min(r).magnitude
            err[m, z, 1] = max(r).magnitude + values[m, z]
    plot_param_per_metric_and_z(values, err)
    plot_bool_indicator_per_metric_and_z(
        sp.mean(results_for_exp_inftau, axis=2) > mark)


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
        return 3, len(cfg['experiments']), \
            row * len(cfg['experiments']) + experiment_idx + 1

    for e, experiment in enumerate(cfg['experiments']):
            def set_rowlabels(rowname):
                if e <= 0:
                    plt.ylabel(rowname)
                else:
                    plt.yticks([])

            plt.subplot(*subplot_args(e, 0))
            plt.title(experiment['name'])

            set_rowlabels(r"$\langle I^* \rangle$")
            plt.xticks([])
            plt.xlim(0, (len(cfg['metrics']) + 0.5) * (len(cfg['zs']) + 2))
            plot_optimal_uncertainty_reduction(results[e], results_inftau[e])

            plt.subplot(*subplot_args(e, 1))
            plt.ylim([cfg['time_scales'][0], 3 * cfg['time_scales'][-1]])
            plt.semilogy()
            set_rowlabels(r"$\langle \tau^* \rangle$")
            plt.xticks([])
            plt.xlim(0, (len(cfg['metrics']) + 0.5) * (len(cfg['zs']) + 2))
            plot_optimal_tau(results[e], results_inftau[e])

            plt.subplot(*subplot_args(e, 2))
            plt.ylim([cfg['time_scales'][0], 3 * cfg['time_scales'][-1]])
            plt.semilogy()
            set_rowlabels(r"$\tau^*_{\langle I \rangle}$")
            plt.xlim(0, (len(cfg['metrics']) + 0.5) * (len(cfg['zs']) + 2))
            plot_optimal_tau_for_mean_uncertainty_reduction(
                results[e], results_inftau[e])

            plt.xticks(
                (sp.arange(len(cfg['metrics'])) + 0.5) * (len(cfg['zs']) + 2),
                ['$%s$' % m for m in cfg['metrics']])


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
            'plot_uncertainty_reduction': list,
            'zs': list,
            'plot_z_curves': list,
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
