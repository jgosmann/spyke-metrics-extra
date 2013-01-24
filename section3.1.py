#!/usr/bin/env python

from __future__ import division
import quantities as pq
import scipy as sp
import spykeutils.spike_train_generation as stg

evaluation_points = 50
max_rate = 50 * pq.Hz
spike_trains_per_rate = 50
spike_train_length = 3 * pq.s

trains = {}
for rate in sp.linspace(1 * pq.Hz, max_rate, evaluation_points):
    trains[rate] = [stg.gen_homogeneous_poisson(
        rate, t_stop=spike_train_length) for i in xrange(spike_trains_per_rate)]
