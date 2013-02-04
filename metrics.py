import scipy as sp
import spykeutils.signal_processing as sigproc
import spykeutils.spike_train_metrics as stm
import spykeutils.tools as stools


def binning_distance(trains, tau, exponent=2):
    if sp.isinf(tau):
        num_spikes = sp.atleast_2d([st.size for st in trains])
        return sp.absolute(num_spikes.T - num_spikes)
    sampling_rate = 1.0 / tau
    binned, dummy = stools.bin_spike_trains({0: trains}, sampling_rate)
    binned = sp.tile(sp.vstack(binned[0]), (len(trains), 1, 1))
    return sp.asfarray(sp.sum(sp.absolute(
        binned - sp.rollaxis(binned, 1)) ** exponent, axis=2))


def normalized_vp_dist(trains, tau):
    num_spikes = sp.atleast_2d(sp.asarray([st.size for st in trains]))
    normalization = num_spikes + num_spikes.T
    normalization[normalization == 0.0] = 1.0
    return stm.victor_purpura_dist(trains, 2.0 / tau, sort=False) / normalization


metrics = {
    'D_V': lambda trains, tau: stm.victor_purpura_dist(
        trains, 2.0 / tau, sort=False),
    'D_V^*': normalized_vp_dist,
    'D_R': lambda trains, tau: stm.van_rossum_dist(
        trains, tau, sort=False) ** 2 / 2.0,
    'D_{R*}': lambda trains, tau: stm.van_rossum_dist(
        trains, kernel=sigproc.TriangularKernel(tau, normalize=False),
        sort=False),
    'D_B': binning_distance,
    'D_{B*}': lambda trains, tau: binning_distance(trains, tau, 1),
    'D_{ES}': lambda trains, tau: stm.event_synchronization(
        trains, tau, sort=False),
    'D_{HM}': stm.hunter_milton_similarity
}
