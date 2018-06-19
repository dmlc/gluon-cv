"""Plot History from Training"""
import os


class TrainingHistory(object):
    r"""Training History Record and Plot

    Parameters
    ----------
    labels : list of str
        List of names of the labels in the history.
    """
    def __init__(self, labels):
        self.l = len(labels)
        self.history = {}
        self.epochs = 0
        self.labels = labels
        for lb in self.labels:
            self.history[lb] = []

    def update(self, values):
        r"""Update the training history

        Parameters
        ---------
        values: list of float
            List of metric scores for each label.
        """
        assert len(values) == self.l
        for i, v in enumerate(values):
            label = self.labels[i]
            self.history[label].append(v)
        self.epochs += 1

    def plot(self, labels=None, colors=None, y_lim=(0, 1),
             save_path=None, legend_loc='upper right'):
        r"""Update the training history

        Parameters
        ---------
        labels: list of str
            List of label names to plot.
        colors: list of str
            List of line colors.
        save_path: str
            Path to save the plot. Will plot to screen if is None.
        legend_loc: str
            location of legend. upper right by default.
        """
        import matplotlib.pyplot as plt

        if labels is None:
            labels = self.labels
        n = len(labels)

        line_lists = [None]*n
        if colors is None:
            colors = ['C'+str(i) for i in range(n)]
        else:
            assert len(colors) == n

        plt.ylim(y_lim)
        for i, lb in enumerate(labels):
            line_lists[i], = plt.plot(list(range(self.epochs)),
                                      self.history[lb],
                                      colors[i],
                                      label=lb)
        plt.legend(tuple(line_lists), labels, loc=legend_loc)
        if save_path is None:
            plt.show()
        else:
            save_path = os.path.expanduser(save_path)
            plt.savefig(save_path)
