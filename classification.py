from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from spykeutils.plugin import analysis_plugin, gui_data
import itertools
import matplotlib.pyplot as plt
import quantities as pq
import numpy as np
import scipy as sp
from spykeutils import SpykeException
from spykeutils.sklearn_bindings import metric_defs
from spykeutils.sklearn_bindings import PrecomputedSpikeTrainMetricApplier


class SvmClassifierPlugin(analysis_plugin.AnalysisPlugin):
    metrics_to_use = gui_data.MultipleChoiceItem(
        "Metrics", zip(
            metric_defs.keys(),
            (name for name, unused in metric_defs.itervalues())))
    c_values_expr = gui_data.StringItem(
        "C values", "sp.logspace(-4, 4, 9)", notempty=True)
    tau_values_expr = gui_data.StringItem(
        "tau values", "sp.logspace(0, 2, 5) * pq.ms", notempty=True)
    n_jobs = gui_data.IntItem("Number of parallel jobs", default=1, min=1)
    verbosity = gui_data.IntItem("Verbosity", default=2, min=0)

    metric_param_prefix = 'metric'
    svm_param_prefix = 'svm'
    metric_key = metric_param_prefix + '__metric'
    tau_key = metric_param_prefix + '__tau'
    c_key = svm_param_prefix + '__C'

    eval_env = {
        'np': np, 'numpy': np,
        'pq': pq, 'quantities': pq,
        'sp': sp, 'scipy': sp
    }

    def get_name(self):
        return 'Classify spike trains'

    def start(self, current, selections):
        self.check_config()

        current.progress.begin()
        current.progress.setLabelText("Grid search ...")

        trains = list(itertools.chain(
            *(selection.spike_trains() for selection in selections)))
        targets = sp.array(list(itertools.chain(
            *([i] * len(selection.spike_trains())
              for i, selection in enumerate(selections)))))

        metricApplier = PrecomputedSpikeTrainMetricApplier(
            trains, self.metrics_to_use[0])
        pipe = Pipeline(steps=[
            (self.metric_param_prefix, metricApplier),
            (self.svm_param_prefix, svm.SVC(kernel='precomputed'))])

        clf = GridSearchCV(
            pipe, self.get_param_grid(), n_jobs=self.n_jobs,
            verbose=self.verbosity)
        clf.fit(metricApplier.x_in, targets)
        current.progress.done()

        print "Best score %.3f with parameters %s." % \
            (clf.best_score_, str(clf.best_estimator_.get_params()))
        self.plot_gridsearch_scores_per_metric(clf.grid_scores_)
        plt.show()

    def check_config(self):
        if self.n_jobs != 1 and self.verbosity > 0:
            raise SpykeException(
                "Verbose output is not possible with more than one job.")

    def get_param_grid(self):
        return {
            self.metric_key: self.metrics_to_use,
            self.tau_key: eval(self.tau_values_expr, self.eval_env),
            self.c_key: eval(self.c_values_expr, self.eval_env)}

    def plot_gridsearch_scores_per_metric(self, grid_scores):
        cols = int(sp.ceil(sp.sqrt(len(self.metrics_to_use))))
        rows = int(sp.ceil(len(self.metrics_to_use) / cols))
        for i, metric in enumerate(self.metrics_to_use):
            plt.subplot(rows, cols, i + 1)
            self.plot_gridsearch_scores(grid_scores, {self.metric_key: metric})
            plt.title(metric_defs[metric][0])

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
        cb = plt.colorbar()
        cb.set_label("Accuracy")
