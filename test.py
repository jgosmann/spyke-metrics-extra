from sklearn import svm
from spykeutils.plugin import analysis_plugin
import itertools
import quantities as pq
import spykeutils.spike_train_metrics as stm


class SvmSpikeTrainClassifier(svm.SVC):
    def __init__(self, metric, tau, **kwargs):
        svm.SVC.__init__(self, kernel='precomputed', **kwargs)
        self.metric = metric
        self.tau = tau

    def fit(self, x, y):
        gram = self.metric(x, self.tau)
        return svm.SVC.fit(self, gram, y)

    def predict(self, x):
        gram = self.metric(x, self.tau)
        return svm.SVC.predict(self, gram)


class SvmClassifierPlugin(analysis_plugin.AnalysisPlugin):
    def get_name(self):
        return 'Test plugin'

    def start(self, current, selections):
        trains = list(itertools.chain(
            *(selection.spike_trains() for selection in selections)))
        targets = list(itertools.chain(
            *([i] * len(selection.spike_trains())
              for i, selection in enumerate(selections))))

        clf = SvmSpikeTrainClassifier(stm.van_rossum_dist, 1.0 * pq.s)
        clf.fit(trains, targets)
        print clf.predict(trains)
