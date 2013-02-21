from functools import wraps
from sklearn import cross_validation
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from spykeutils import rate_estimation
from spykeutils.plugin import analysis_plugin, gui_data
from spykeutils.sklearn_bindings import PrecomputedVRMultiunitDistApplier
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


class VRMultiunitClassifierPlugin(analysis_plugin.AnalysisPlugin):
    c_values_expr = gui_data.StringItem(
        "C values", "sp.logspace(-4, 4, 9)", notempty=True)
    tau_values_expr = gui_data.StringItem(
        "tau values", "sp.logspace(0, 2, 5) * pq.ms", notempty=True)
    n_folds = gui_data.IntItem("Crossvalidation folds", default=3, min=2)
    align = gui_data.StringItem('Event label for alignment', default='lastSt')
    n_jobs = gui_data.IntItem("Number of parallel jobs", default=1, min=1)
    output_file = gui_data.FileSaveItem(
        "Save grid search plot to", '*', default='grid_search.pdf')

    metric_param_prefix = 'metric'
    svm_param_prefix = 'svm'
    tau_key = metric_param_prefix + '__tau'
    c_key = svm_param_prefix + '__C'

    eval_env = {
        'np': np, 'numpy': np,
        'pq': pq, 'quantities': pq,
        'sp': sp, 'scipy': sp
    }

    def get_name(self):
        return 'Classify spike trains with multi-unit VR distance'

    def start(self, current, selections):
        current.progress.begin(self.get_name())
        param_grid = self.get_param_grid()
        current.progress.set_ticks(int(
            self.n_folds * len(param_grid[self.tau_key]) *
            len(param_grid[self.c_key])))

        print "Align spike trains"
        # FIXME: This needs to be done completely different as we need the
        # spike trains per unit which is a dict and not a list. Then the lists
        # in the dict have to be split into the individual sets for training,
        # test, cross-validation and so on. But because the lists are values in
        # a dict it cannot be done automatically by scikit learn.
        trains = []
        targets = []
        for i, s in enumerate(selections):
            events = s.labeled_events(self.align)
            for seg in events:  # Align on first event in each segment
                events[seg] = events[seg][0]
            aligned_trains = rate_estimation.aligned_spike_trains(
                s.spike_trains(), events)
            trains.extend(aligned_trains)
            targets.extend([i] * len(s.spike_trains()))

        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(
            target=self._run, args=(child_conn, trains, targets))
        print "Starting grid search"
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
        report = parent_conn.recv()
        confmat = parent_conn.recv()
        parent_conn.close()
        p.join()

        print "Best score %.3f (tau=%s, C=%f)." % \
            (best_score, str(best_params[self.tau_key]),
             best_params[self.c_key])
        print report
        print "Confusion matrix:"
        print confmat
        self.plot_gridsearch_scores(grid_scores)

        current.progress.done()
        plt.savefig(self.output_file)

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

        metricApplier = PrecomputedVRMultiunitDistApplier(trains, 0.5)
        pipe = Pipeline(steps=[
            (self.metric_param_prefix, metricApplier),
            (self.svm_param_prefix,
             svm.SVC(kernel='precomputed', class_weight='auto'))])

        x_train, x_test, targets_train, targets_test = \
            cross_validation.train_test_split(metricApplier.x_in, targets)

        clf = GridSearchCV(
            pipe, self.get_param_grid(), n_jobs=self.n_jobs, cv=self.n_folds,
            verbose=2)
        clf.fit(x_train, targets_train)
        prediction = clf.best_estimator_.predict(x_test)

        connection.send(clf.grid_scores_)
        connection.send(clf.best_score_)
        connection.send(clf.best_estimator_.get_params())
        connection.send(classification_report(targets_test, prediction))
        connection.send(confusion_matrix(targets_test, prediction))
        connection.close()

        # Restore streams
        sys.stdout, sys.stderr = old_streams

    def get_param_grid(self):
        return {
            self.tau_key: eval(self.tau_values_expr, self.eval_env),
            self.c_key: eval(self.c_values_expr, self.eval_env)}

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
        plt.title("VR multi-unit")
        plt.xlabel(self.tau_key)
        plt.ylabel(self.c_key)
        cb = plt.colorbar()
        cb.set_label("Accuracy")
