Profiling and Benchmarking
==========================

The `profile` folder provides some scripts to profile and benchmark the spike
train metrics implemented in spykeutils:

* `profile_spike_train_metrics.py` profiles the metrics and prints a summary.
* `benchmark_spike_train_metrics.py` produces a plot for each metric of the
  calculation time in dependence of the number of spike trains and the number of
  spikes per spike train.
* `benchmark_multiunit_metrics.py` produces a plot for each multi-unit metric
  of the calculation time in dependence of the number of spike trains, the
  number of spike per spike train and the number of units.

The profile `folder` contains also sample plots from the benchmark scripts
produced on a Macbook Pro with a Intel Core 2 Duo 2.4 GHz processor and 4 GB
RAM. A sample profiling result from the same machine is found below:


    Tue Feb 12 15:12:31 2013 /var/folders/71/q23xbk51711f72kl0y4fy1c80000gn/T/tmpNqm0nw

             1295896 function calls (1275062 primitive calls) in 353.184 seconds

       Ordered by: cumulative time
       List reduced from 267 to 10 due to restriction <'^spike_train_metrics.py:\\d+\\([^_<].*(?<!compute)\\)'>

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.000    0.000  352.252  352.252 spike_train_metrics.py:654(victor_purpura_multiunit_dist)
            2    0.000    0.000    0.350    0.175 spike_train_metrics.py:313(st_inner)
            1    0.000    0.000    0.267    0.267 spike_train_metrics.py:531(victor_purpura_dist)
            1    0.000    0.000    0.181    0.181 spike_train_metrics.py:221(norm_dist)
            1    0.000    0.000    0.172    0.172 spike_train_metrics.py:50(cs_dist)
            1    0.000    0.000    0.161    0.161 spike_train_metrics.py:263(schreiber_similarity)
            1    0.000    0.000    0.074    0.074 spike_train_metrics.py:168(hunter_milton_similarity)
            1    0.000    0.000    0.033    0.033 spike_train_metrics.py:460(van_rossum_multiunit_dist)
            1    0.000    0.000    0.030    0.030 spike_train_metrics.py:99(event_synchronization)
            1    0.000    0.000    0.018    0.018 spike_train_metrics.py:410(van_rossum_dist)
