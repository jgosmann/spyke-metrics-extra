from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from spykeutils.plugin import analysis_plugin, gui_data
import itertools
import matplotlib.pyplot as plt
import quantities as pq
import scipy as sp
import sklearn.base
import sklearn.cross_validation
import sklearn.grid_search
import spykeutils.spike_train_metrics as stm


metric_funcs = {
    'event_synchronization':
    lambda trains, tau: stm.event_synchronization(trains, tau),

    'van Rossum distance':
    lambda trains, tau: stm.van_rossum_dist(trains, tau),

    'Victor Purpura\'s distance':
    lambda trains, tau: stm.victor_purpura_dist(trains, 2.0 / tau)
}


class PrecomputedSpikeTrainMetricApplier(
        sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, data, metric=stm.van_rossum_dist, tau=1.0 * pq.s):
        self.metric = metric
        self.tau = tau
        self.data = data
        self.x_in = sp.arange(len(data))

    def fit(self, x, y):
        self.gram = metric_funcs[self.metric](self.data, self.tau)
        self.x_train = x
        return self

    def transform(self, x):
        return self.gram[sp.meshgrid(self.x_train, x)]


class SvmClassifierPlugin(analysis_plugin.AnalysisPlugin):
    metrics_to_use = gui_data.MultipleChoiceItem(
        "Metrics", zip(metric_funcs.keys(), metric_funcs.keys()))
    n_jobs = gui_data.IntItem("Number of parallel jobs", default=1, min=1)

    metric_param_prefix = 'metric'
    svm_param_prefix = 'svm'
    metric_key = metric_param_prefix + '__metric'
    tau_key = metric_param_prefix + '__tau'
    c_key = svm_param_prefix + '__C'

    def get_name(self):
        return 'Classify spike trains'

    def start(self, current, selections):
        trains = list(itertools.chain(
            *(selection.spike_trains() for selection in selections)))
        targets = sp.array(list(itertools.chain(
            *([i] * len(selection.spike_trains())
              for i, selection in enumerate(selections)))))

        metricApplier = PrecomputedSpikeTrainMetricApplier(trains)
        pipe = Pipeline(steps=[
            (self.metric_param_prefix, metricApplier),
            (self.svm_param_prefix, svm.SVC(kernel='precomputed'))])

        clf = GridSearchCV(
            pipe, self.get_param_grid(), n_jobs=self.n_jobs, verbose=1)
        clf.fit(metricApplier.x_in, targets)
        print clf.best_score_, clf.best_estimator_.get_params()
        self.plot_gridsearch_scores_per_metric(clf.grid_scores_)
        plt.show()

    def get_param_grid(self):
        return {
            self.metric_key: self.metrics_to_use,
            self.tau_key: [1, 2, 5, 10, 100] * pq.ms,
            self.c_key: [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    def plot_gridsearch_scores_per_metric(self, grid_scores):
        cols = int(sp.ceil(sp.sqrt(len(self.metrics_to_use))))
        rows = int(sp.ceil(len(self.metrics_to_use) / cols))
        for i, metric in enumerate(self.metrics_to_use):
            plt.subplot(rows, cols, i + 1)
            self.plot_gridsearch_scores(grid_scores, {self.metric_key: metric})
            plt.title(metric)

    def plot_gridsearch_scores(self, grid_scores, filters={}):
        param_grid = self.get_param_grid()
        x, y = sp.meshgrid(param_grid[self.tau_key], param_grid[self.c_key])
        counts = sp.zeros_like(x)
        scores = sp.zeros_like(x)

        for params, s, unused in grid_scores:
            if all(params[key] == filters[key] for key in filters):
                i = sp.searchsorted(param_grid[self.c_key], params[self.c_key])
                j = sp.searchsorted(
                    param_grid[self.tau_key], params[self.tau_key])
                counts[i, j] += 1
                scores[i, j] += s
        scores /= counts

        plt.contourf(x, y, scores)
        plt.loglog()
        plt.xlabel(self.c_key)
        plt.ylabel(self.tau_key)
        plt.colorbar()
