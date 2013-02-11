#!/usr/bin/env python

import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import pstats
import quantities as pq
import scipy as sp
import spykeutils.signal_processing as sigproc
import spykeutils.spike_train_generation as stg
import spykeutils.spike_train_metrics as stm
import sys
import timeit


tau = 5.0 * pq.ms
sampling_rate = 1000 * pq.Hz


def trains_as_multiunits(trains):
    half = len(trains) // 2
    return {0: trains[:half], 1: trains[half:2 * half]}


metrics = {
    'cs': ('Cauchy-Schwarz distance',
           lambda trains: stm.cs_dist(
               trains, sigproc.CausalDecayingExpKernel(tau), sampling_rate)),
    'es': ('event synchronization',
           lambda trains: stm.event_synchronization(trains, tau, sort=False)),
    'hm': ('Hunter-Milton similarity measure',
           lambda trains: stm.hunter_milton_similarity(trains, tau)),
    'norm': ('norm distance',
             lambda trains: stm.norm_dist(
                 trains, sigproc.CausalDecayingExpKernel(tau), sampling_rate)),
    'ss': ('Schreiber et al. similarity measure',
           lambda trains: stm.schreiber_similarity(
               trains, sigproc.GaussianKernel(tau), sort=False)),
    'vr': ('van Rossum distance',
           lambda trains: stm.van_rossum_dist(trains, tau, sort=False)),
    'vp': ('Victor-Purpura\'s distance',
           lambda trains: stm.victor_purpura_dist(trains, 2.0 / tau)),
    'vr_mu': ('van Rossum multi-unit distance',
              lambda trains: stm.van_rossum_multiunit_dist(
                  trains_as_multiunits(trains), 0.5, tau)),
    'vp_mu': ('Victor-Purpura\'s multi-unit distance',
              lambda trains: stm.victor_purpura_multiunit_dist(
                  trains_as_multiunits(trains), 0.3, 2.0 / tau))}


def print_available_metrics():
    for key in metrics:
        print "%s  (%s)" % (key, metrics[key][0])


def print_summary(profile_file):
    stats = pstats.Stats(profile_file)
    stats.strip_dirs().sort_stats('cumulative').print_stats(
        r'^spike_train_metrics.py:\d+\([^_<].*(?<!compute)\)')


def profile_metrics(trains, to_profile):
    for key in to_profile:
        metrics[key][1](trains)


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

    def benchmark_metric(self, metric):
        times = sp.empty((
            len(self.data.spike_count_range), len(self.data.train_count_range)))
        for i, j in sp.ndindex(*times.shape):
            trains = self.data.trains[i][:self.data.train_count_range[j]]
            assert trains is not None  # Suppress pyflakes unused warning
            times[i, j] = timeit.timeit(
                lambda: metric(trains), number=self.num_loops)
        return times


def plot_benchmark(times, data):
    left = data.train_count_range[0]
    right = data.train_count_range[-1]
    bottom = data.spike_count_range[0]
    top = data.spike_count_range[-1]
    plt.imshow(
        times, origin='lower', aspect='auto', extent=(left, right, bottom, top),
        cmap=cm.get_cmap('hot'))
    plt.xlabel("# spike trains")
    plt.ylabel("# spikes per spike trains")
    cb = plt.colorbar()
    cb.set_label("Calculation time / s")


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
    args = parser.parse_args()

    if args.list_metrics:
        print_available_metrics()
        sys.exit(0)

    try:
        with open(args.data[0], 'r') as f:
            data = pickle.load(f)
        print "Loaded stored benchmarking data."
    except:
        data = BenchmarkData(sp.arange(1, 200, 10), sp.arange(10))
        if args.data is not None:
            with open(args.data[0], 'w') as f:
                pickle.dump(data, f)
            print "Stored benchmarking data."

    b = Benchmark(data, 2)
    plt.figure()
    plot_benchmark(b.benchmark_metric(metrics['vr'][1]), data)
    plt.show()
