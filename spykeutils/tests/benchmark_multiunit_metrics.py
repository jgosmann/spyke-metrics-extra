#!/usr/bin/env python

import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import quantities as pq
import scipy as sp
import spykeutils.spike_train_generation as stg
import spykeutils.spike_train_metrics as stm
import sys
import timeit


tau = 5.0 * pq.ms


def trains_as_multiunits(trains, num_units):
    trains_per_unit = len(trains) // num_units
    units = {}
    for i in xrange(num_units):
        units[i] = trains[i * trains_per_unit:(i + 1) * trains_per_unit]
    return units


metrics = {
    'vp': ('Victor Purpursa\'s distance',
           lambda units: stm.victor_purpura_multiunit_dist(
               units, 0.7, 2.0 / tau)),
    'vr': ('van Rossum distance',
           lambda units: stm.van_rossum_multiunit_dist(units, 0.5, tau))}


def print_available_metrics():
    for key in metrics:
        print "%s  (%s)" % (key, metrics[key][0])


class BenchmarkData(object):
    def __init__(
            self, spike_count_range, train_count_range, num_units_range,
            firing_rate=50 * pq.Hz):
        self.spike_count_range = spike_count_range
        self.train_count_range = train_count_range
        self.num_units_range = num_units_range
        self.num_trains_per_spike_count = sp.amax(train_count_range)
        self.trains = [
            [stg.gen_homogeneous_poisson(firing_rate, max_spikes=num_spikes)
             for i in xrange(self.num_trains_per_spike_count)]
            for num_spikes in spike_count_range]


class Benchmark(object):
    def __init__(self, benchmark_data, num_loops=10):
        self.num_loops = num_loops
        self.data = benchmark_data
        self.results = {}

    def benchmark_single_metric(self, metric):
        print "Benchmarking %s metric ..." % metric
        times = sp.empty((
            len(self.data.num_units_range), len(self.data.spike_count_range),
            len(self.data.train_count_range)))
        for u, i, j in sp.ndindex(*times.shape):
            units = trains_as_multiunits(
                self.data.trains[i][:self.data.train_count_range[j]],
                self.data.num_units_range[u])
            times[u, i, j] = timeit.timeit(
                lambda: metrics[metric][1](units), number=self.num_loops)
        self.results[metric] = times

    def benchmark_metrics(self, to_benchmark):
        for m in to_benchmark:
            self.benchmark_single_metric(m)

    def plot_times(self, times):
        left = self.data.train_count_range[0]
        right = self.data.train_count_range[-1]
        bottom = self.data.spike_count_range[0]
        top = self.data.spike_count_range[-1]
        plt.imshow(
            times, origin='lower', aspect='auto',
            extent=(left, right, bottom, top), cmap=cm.get_cmap('hot'))
        plt.xlabel("# spike trains")
        plt.ylabel("# spikes per spike trains")
        cb = plt.colorbar()
        cb.set_label("Calculation time / s")

    def plot_benchmark_results(self):
        plt.figure()
        rows = len(self.data.num_units_range)
        cols = len(self.results)
        for i, m in enumerate(self.results):
            for j, u in enumerate(self.data.num_units_range):
                plt.subplot(rows, cols, i + j * cols + 1)
                self.plot_times(self.results[m][j])
                if j <= 0:
                    plt.title(m)
                if i <= 0:
                    plt.figtext(
                        0.02, 0.8 / rows * j + 0.2, "%i units" % u,
                        rotation='vertical')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Profile the spike train distances.")
    parser.add_argument(
        '--data', '-d', type=str, nargs=1,
        help="Use given file to load spike trains for benchmarking. If not " +
        "existent, spike trains will be generated as usual and saved in this " +
        "file.")
    parser.add_argument(
        '--metrics', '-m', type=str, nargs='*', default=metrics.iterkeys(),
        choices=metrics.keys(),
        help="Spike train metrics to profile. Defaults to all available " +
        "metrics.")
    parser.add_argument(
        '--list-metrics', '-l', action='store_const', default=False,
        const=True,
        help="Print a list of spike train metrics which can be used with " +
        "the -m option.")
    parser.add_argument(
        '--output', '-o', type=str, nargs=1, help="Save plot to given file.")
    parser.add_argument(
        '--show', '-s', action='store_const', default=False, const=True,
        help="Show plot after benchmarking.")
    args = parser.parse_args()

    if args.list_metrics:
        print_available_metrics()
        sys.exit(0)

    try:
        with open(args.data[0], 'r') as f:
            data = pickle.load(f)
        print "Loaded stored benchmarking data."
    except:
        data = BenchmarkData(
            sp.arange(10, 210, 10), sp.arange(2, 11), sp.arange(2, 12, 2))
        if args.data is not None:
            with open(args.data[0], 'w') as f:
                pickle.dump(data, f)
            print "Stored benchmarking data."

    b = Benchmark(data)
    b.benchmark_metrics(args.metrics)
    b.plot_benchmark_results()

    if args.output is not None:
        plt.savefig(args.output[0])
        if args.show:
            plt.show()
    else:
        plt.show()
