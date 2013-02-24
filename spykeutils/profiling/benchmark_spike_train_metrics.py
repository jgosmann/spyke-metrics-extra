#!/usr/bin/env python

import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import quantities as pq
import scipy as sp
import spykeutils.signal_processing as sigproc
import spykeutils.spike_train_generation as stg
import spykeutils.spike_train_metrics as stm
import sys
import timeit


tau = 5.0 * pq.ms
sampling_rate = 1000 * pq.Hz


metrics = {
    'cs': (r'$D_{\mathrm{CS}}$',
           lambda trains: stm.cs_dist(
               trains, sigproc.CausalDecayingExpKernel(tau), sampling_rate)),
    'es': (r'$S_{\mathrm{ES}}$',
           lambda trains: stm.event_synchronization(trains, tau, sort=False)),
    'hm': (r'$S_{\mathrm{HM}}$',
           lambda trains: stm.hunter_milton_similarity(trains, tau)),
    'norm': (r'$D_{\mathrm{ND}}$',
             lambda trains: stm.norm_dist(
                 trains, sigproc.CausalDecayingExpKernel(tau), sampling_rate)),
    'ss': (r'$S_{S}}$',
           lambda trains: stm.schreiber_similarity(
               trains, sigproc.LaplacianKernel(tau), sort=False)),
    'vr': (r'$D_{\mathrm{R}}$',
           lambda trains: stm.van_rossum_dist(trains, tau, sort=False)),
    'vp': (r'$D_{\mathrm{V}}$',
           lambda trains: stm.victor_purpura_dist(trains, 2.0 / tau))}


def print_available_metrics():
    for key in metrics:
        print "%s  (%s)" % (key, metrics[key][0])


class BenchmarkData(object):
    def __init__(
            self, spike_count_range, train_count_range, firing_rate=50 * pq.Hz):
        self.spike_count_range = spike_count_range
        self.train_count_range = train_count_range
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
            len(self.data.spike_count_range), len(self.data.train_count_range)))
        for i, j in sp.ndindex(*times.shape):
            trains = self.data.trains[i][:self.data.train_count_range[j]]
            times[i, j] = timeit.timeit(
                lambda: metrics[metric][1](trains), number=self.num_loops) / \
                self.num_loops
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
        cols = int(sp.ceil(sp.sqrt(len(self.results))))
        rows = int(sp.ceil(float(len(self.results)) / cols))
        for i, m in enumerate(self.results):
            plt.subplot(rows, cols, i + 1)
            self.plot_times(self.results[m])
            plt.title(metrics[m][0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create calculation time plots of different spike train " +
        "metrics.")
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
        data = BenchmarkData(sp.arange(10, 210, 10), sp.arange(2, 11))
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
