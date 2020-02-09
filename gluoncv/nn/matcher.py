# pylint: disable=arguments-differ
"""Matchers for target assignment.
Matchers are commonly used in object-detection for anchor-groundtruth matching.
The matching process is a prerequisite to training target assignment.
Matching is usually not required during testing.
"""
from __future__ import absolute_import
from mxnet import gluon
from mxnet.gluon import nn


class CompositeMatcher(gluon.HybridBlock):
    """A Matcher that combines multiple strategies.

    Parameters
    ----------
    matchers : list of Matcher
        Matcher is a Block/HybridBlock used to match two groups of boxes
    """
    def __init__(self, matchers):
        super(CompositeMatcher, self).__init__()
        assert len(matchers) > 0, "At least one matcher required."
        for matcher in matchers:
            assert isinstance(matcher, (gluon.Block, gluon.HybridBlock))
        self._matchers = nn.HybridSequential()
        for m in matchers:
            self._matchers.add(m)

    def hybrid_forward(self, F, x):
        matches = [matcher(x) for matcher in self._matchers]
        return self._compose_matches(F, matches)

    def _compose_matches(self, F, matches):
        """Given multiple match results, compose the final match results.
        The order of matches matters. Only the unmatched(-1s) in the current
        state will be substituted with the matching in the rest matches.

        Parameters
        ----------
        matches : list of NDArrays
            N match results, each is an output of a different Matcher

        Returns
        -------
         one match results as (B, N, M) NDArray
        """
        result = matches[0]
        for match in matches[1:]:
            result = F.where(result > -0.5, result, match)
        return result


class BipartiteMatcher(gluon.HybridBlock):
    """A Matcher implementing bipartite matching strategy.

    Parameters
    ----------
    threshold : float
        Threshold used to ignore invalid paddings
    is_ascend : bool
        Whether sort matching order in ascending order. Default is False.
    eps : float
        Epsilon for floating number comparison
    share_max : bool, default is True
        The maximum overlap between anchor/gt is shared by multiple ground truths.
        We recommend Fast(er)-RCNN series to use ``True``, while for SSD, it should
        defaults to ``False`` for better result.
    """
    def __init__(self, threshold=1e-12, is_ascend=False, eps=1e-12, share_max=True):
        super(BipartiteMatcher, self).__init__()
        self._threshold = threshold
        self._is_ascend = is_ascend
        self._eps = eps
        self._share_max = share_max

    def hybrid_forward(self, F, x):
        """BipartiteMatching

        Parameters:
        ----------
        x : NDArray or Symbol
            IOU overlaps with shape (N, M), batching is supported.
        ###############################
        can use this array for test:
        ious = nd.array([
       [[0.56910074, 0.13997258, 0.4071833,  0., 0.],
        [0.03322238, 0.06916699, 0.98257494, 0., 0.],
        [0.69742876, 0.37329075, 0.45354268, 0., 0.],
        [0.42007536, 0.7220556 , 0.05058811, 0., 0.],
        [0.8663823 , 0.3654961 , 0.9755215 , 0., 0.]],

       [[0.01662797, 0.8558034 , 0.23074234, 0., 0.],
        [0.01171408, 0.7649117 , 0.35997805 ,0., 0.],
        [0.9441235,  0.72999054, 0.7499992 , 0., 0.],
        [0.17162968, 0.3394038 , 0.5210366 , 0., 0.],
        [0.48954895, 0.05433799, 0.33898512, 0., 0.]]])

        """
        # bipartite_matching returns the no-shared overlap
        match = F.contrib.bipartite_matching(x, threshold=self._threshold,
                                             is_ascend=self._is_ascend)
        if self._share_max is False:
            # if share_max is False, can directly return
            return match[0]
        
        if self._is_ascend:
            arg_ = F.argmin
            max_min_ = F.min
            comp_ = lambda a, b: a < b + self._eps
        else:
            arg_ = F.argmax
            max_min_ = F.max
            comp_ = lambda a, b: a + self._eps > b
            
        Rargax = arg_(x, axis=-1)
        Rm = max_min_(x, axis=-1)
        # Filter value
        Rargax = F.where(comp_(Rm, self._threshold), 
                             Rargax, F.ones_like(Rargax) * -1)
        # add shared gt index
        result = F.where(comp_(F.pick(x, match[0], axis=-1), Rm), 
                             match[0], Rargax)
        return result

class MaximumMatcher(gluon.HybridBlock):
    """A Matcher implementing maximum matching strategy.

    Parameters
    ----------
    threshold : float
        Matching threshold.

    """
    def __init__(self, threshold):
        super(MaximumMatcher, self).__init__()
        self._threshold = threshold

    def hybrid_forward(self, F, x):
        argmax = F.argmax(x, axis=-1)
        match = F.where(F.pick(x, argmax, axis=-1) >= self._threshold, argmax,
                        F.ones_like(argmax) * -1)
        return match
