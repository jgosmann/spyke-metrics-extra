from functools import wraps
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from spykeutils.plugin import analysis_plugin, gui_data
from spykeutils.sklearn_bindings import metric_defs
from spykeutils.sklearn_bindings import PrecomputedSpikeTrainMetricApplier
import itertools
import matplotlib.pyplot as plt
import multiprocessing
import quantities as pq
import numpy as np
import scipy as sp
import sys


class ConnectionWriter(object):
    """Provides a file-like object for redirecting write operation to
    a :class:`multiprocessing.Connection` object.
    """

    def __init__(self, connection):
        self.connection = connection

    def close(self):
        self.connection.close()

    def fileno(self):
        return self.connection.fileno()

    def flush(self):
        pass

    def write(self, str):
        self.connection.send(str)

    def writelines(self, sequence):
        for str in sequence:
            self.write(str)


class SynchronizedConnection(object):
    """Synchronizes a :class:`multiprocessing.Connection` object to allow access
    from multiple processes.

    Note that :meth:`send_bytes`, :meth:`recv_bytes` and :meth:`recv_bytes_into`
    are not implemented.
    """

    def synchronized(func):
        @wraps(func)
        def _synchronized_function(self, *args, **kwargs):
            self.lock.acquire()
            try:
                return func(self, *args, **kwargs)
            finally:
                self.lock.release()
        return _synchronized_function

    def __init__(self, connection):
        self.connection = connection
        self.lock = multiprocessing.Lock()

    @synchronized
    def send(self, obj):
        self.connection.send(obj)

    @synchronized
    def recv(self):
        return self.connection.recv()

    def fileno(self):
        # no synchronization needed, fileno should be immutable
        return self.fileno()

    @synchronized
    def close(self):
        return self.connection.close()

    @synchronized
    def poll(self, timeout=0):
        return self.connection.poll(timeout)


class SvmClassifierPlugin(analysis_plugin.AnalysisPlugin):
    metrics_to_use = gui_data.MultipleChoiceItem(
        "Metrics", zip(
            metric_defs.keys(),
            (name for name, unused in metric_defs.itervalues())))
    c_values_expr = gui_data.StringItem(
        "C values", "sp.logspace(-4, 4, 9)", notempty=True)
    tau_values_expr = gui_data.StringItem(
        "tau values", "sp.logspace(0, 2, 5) * pq.ms", notempty=True)
    n_folds = gui_data.IntItem("Crossvalidation folds", default=3, min=2)
    n_jobs = gui_data.IntItem("Number of parallel jobs", default=1, min=1)

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
        current.progress.begin()
        param_grid = self.get_param_grid()
        current.progress.set_ticks(int(
            self.n_folds * len(param_grid[self.metric_key]) *
            len(param_grid[self.tau_key]) * len(param_grid[self.c_key])))

        trains = list(itertools.chain(
            *(selection.spike_trains() for selection in selections)))
        targets = sp.array(list(itertools.chain(
            *([i] * len(selection.spike_trains())
              for i, selection in enumerate(selections)))))

        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(
            target=self._run, args=(child_conn, trains, targets))
        p.start()
        while True:
            data = parent_conn.recv()
            if isinstance(data, str):
                self._process_gridsearch_msg(current.progress, data)
            else:
                grid_scores = data
                break
        best_score = parent_conn.recv()
        best_params = parent_conn.recv()
        parent_conn.close()
        p.join()

        print "Best score %.3f with %s (tau=%s, C=%f)." % \
            (best_score, metric_defs[best_params[self.metric_key]][0],
             str(best_params[self.tau_key]), best_params[self.c_key])
        self.plot_gridsearch_scores_per_metric(grid_scores)

        current.progress.done()
        plt.show()

    def _process_gridsearch_msg(self, progress, msg):
        msg = msg.strip()
        if msg.startswith('[GridSearchCV]'):
            if msg.endswith('.'):
                progress.step()
                progress.set_status(msg)
        elif msg != "":
            print msg

    def _run(self, connection, trains, targets):
        # Setup synchronized stdout and stderr because the grid search jobs may
        # write to those streams asynchronously. In combination with the GUI
        # this would lead to a crash.
        old_streams = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = ConnectionWriter(
            SynchronizedConnection(connection))

        metricApplier = PrecomputedSpikeTrainMetricApplier(
            trains, self.metrics_to_use[0])
        pipe = Pipeline(steps=[
            (self.metric_param_prefix, metricApplier),
            (self.svm_param_prefix, svm.SVC(kernel='precomputed'))])

        clf = GridSearchCV(
            pipe, self.get_param_grid(), n_jobs=self.n_jobs, cv=self.n_folds,
            verbose=2)
        clf.fit(metricApplier.x_in, targets)

        connection.send(clf.grid_scores_)
        connection.send(clf.best_score_)
        connection.send(clf.best_estimator_.get_params())
        connection.close()

        # Restore streams
        sys.stdout, sys.stderr = old_streams

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
