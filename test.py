from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from spykeutils.plugin import analysis_plugin
import itertools
import quantities as pq
import scipy as sp
import sklearn.base
import sklearn.cross_validation
import sklearn.grid_search
import spykeutils.spike_train_metrics as stm


class PrecomputedSpikeTrainMetricApplier(
        sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, data, metric=stm.van_rossum_dist, tau=1.0 * pq.s):
        self.metric = metric
        self.tau = tau
        self.data = data
        self.x_in = sp.arange(len(data))

    def fit(self, x, y):
        self.gram = self.metric(self.data, self.tau)
        self.x_train = x
        return self

    def transform(self, x):
        return self.gram[sp.meshgrid(self.x_train, x)]


class SvmClassifierPlugin(analysis_plugin.AnalysisPlugin):
    def get_name(self):
        return 'Test plugin'

    def start(self, current, selections):
        trains = list(itertools.chain(
            *(selection.spike_trains() for selection in selections)))
        targets = sp.array(list(itertools.chain(
            *([i] * len(selection.spike_trains())
              for i, selection in enumerate(selections)))))

        metricApplier = PrecomputedSpikeTrainMetricApplier(trains)
        pipe = Pipeline(steps=[
            ('metric', metricApplier),
            ('svm', svm.SVC(kernel='precomputed'))])

        param_grid = [{'svm__C': [0.1, 0.5, 1, 10, 100, 1000]}]

        clf = GridSearchCV(pipe, param_grid)
        clf.fit(metricApplier.x_in, targets)
        print clf.best_score_, clf.best_estimator_.get_params()['svm__C']
