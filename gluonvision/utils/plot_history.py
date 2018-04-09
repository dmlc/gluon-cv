"""Plot History from Training"""
import matplotlib.pyplot as plt

class TrainingHistory():
    r"""Poly like Learning Rate Scheduler
    It returns a new learning rate by::

        lr = baselr * (1 - iter/maxiter) ^ power

    Parameters
    ----------
    items : list of str
        List of names of the items in the history
    """
    def __init__(self, items):
        super(TrainingHistory, self).__init__()
        self.history = {}
        if items:
            for it in items:
                self.history[it] = []
        self.l = len(items)
        self.epochs = 0

    def update(self, kv):
        for k,v in kv.items():
            self.history[k].append(v)
        self.epochs += 1

    def plot(self, items, tofile=None, legend_loc='upper right'):
        for it in items:
            assert self.history[it] is not None
            plt.plot(list(range(self.epochs)), self.history[it])
        plt.legend(items, loc=legend_loc)
        if tofile is None:
            plt.show()
        else:
            plt.savefig(tofile)
