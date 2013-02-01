import scipy as sp
import spykeutils.spike_train_metrics as stm
import spykeutils.tools as stools


def binning_distance(trains, tau):
    if sp.isinf(tau):
        num_spikes = sp.atleast_2d([st.size for st in trains])
        return sp.absolute(num_spikes.T - num_spikes)
    sampling_rate = 1.0 / tau
    binned, dummy = stools.bin_spike_trains({0: trains}, sampling_rate)
    binned = sp.tile(sp.vstack(binned[0]), (len(trains), 1, 1))
    return sp.asfarray(sp.sum((binned - sp.rollaxis(binned, 1)) ** 2, axis=2))


def normalized_vp_dist(trains, tau):
    num_spikes = sp.atleast_2d(sp.asarray([st.size for st in trains]))
    return stm.victor_purpura_dist(
        trains, 2.0 / tau, sort=False) / (num_spikes + num_spikes.T)


metrics = {
    'D_V': lambda trains, tau: stm.victor_purpura_dist(
        trains, 2.0 / tau, sort=False),
    'D_V^*': normalized_vp_dist,
    'D_R': lambda trains, tau: stm.van_rossum_dist(trains, tau, sort=False),
    'D_B': binning_distance,
    'D_{ES}': lambda trains, tau: stm.event_synchronization(
        trains, tau, sort=False),
    'D_{HM}': stm.hunter_milton_similarity
}
