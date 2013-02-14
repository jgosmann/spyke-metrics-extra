from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from spykeutils.plugin import analysis_plugin
import itertools
import matplotlib.pyplot as plt
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

        param_grid = {
            'metric__tau': [1, 2, 5, 10, 100] * pq.ms,
            'svm__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        clf = GridSearchCV(pipe, param_grid, verbose=1)
        clf.fit(metricApplier.x_in, targets)
        print clf.best_score_, clf.best_estimator_.get_params()
        self.plot_gridsearch_scores(
            clf.grid_scores_, param_grid, ('svm__C', 'metric__tau'))

    def plot_gridsearch_scores(
            self, grid_scores, param_grid, (x_param, y_param)):
        x, y = sp.meshgrid(param_grid[y_param], param_grid[x_param])
        counts = sp.zeros_like(x)
        scores = sp.zeros_like(x)

        for params, s, unused in grid_scores:
            i = sp.searchsorted(param_grid[x_param], params[x_param])
            j = sp.searchsorted(param_grid[y_param], params[y_param])
            counts[i, j] += 1
            scores[i, j] += s
        scores /= counts

        plt.contourf(x, y, scores)
        plt.loglog()
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.colorbar()
        plt.show()
