# pylint: disable=arguments-differ
"""Matchers for target assignment.
Matchers are commonly used in object-detection for anchor-groundtruth matching.
The matching process is a prerequisite to training target assignment.
Matching is usually not required during testing.
"""
from __future__ import absolute_import
import mxnet as mx
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

        """
        match = F.contrib.bipartite_matching(x.as_nd_ndarray(), threshold=self._threshold,
                                             is_ascend=self._is_ascend)
        # make sure if iou(a, y) == iou(b, y), then b should also be a good match
        # otherwise positive/negative samples are confusing
        # potential argmax and max
        pargmax = x.as_nd_ndarray().argmax(axis=-1, keepdims=True).as_np_ndarray()  # (B, num_anchor, 1)
        maxs = x.max(axis=len(x.shape)-2, keepdims=True)  # (B, 1, num_gt)
        if self._share_max:
            mask = F.broadcast_greater_equal((x + self._eps).as_nd_ndarray(),
                                             maxs.as_nd_ndarray()).as_np_ndarray()  # (B, num_anchor, num_gt)
            mask = F.np.sum(mask, axis=-1, keepdims=True)  # (B, num_anchor, 1)
        else:
            pmax = F.npx.pick(x, pargmax, axis=-1, keepdims=True)   # (B, num_anchor, 1)
            mask = F.broadcast_greater_equal((pmax + self._eps).as_nd_ndarray(),
                                             maxs.as_nd_ndarray()).as_np_ndarray()  # (B, num_anchor, num_gt)
            mask = F.npx.pick(mask, pargmax, axis=-1, keepdims=True)  # (B, num_anchor, 1)
        new_match = F.where((mask > 0).as_nd_ndarray(), pargmax.as_nd_ndarray(), (F.np.ones_like(pargmax) * -1).as_nd_ndarray())
        result = F.where((match[0] < 0).as_nd_ndarray(), new_match.squeeze(axis=-1), match[0].as_nd_ndarray())
        return result.as_np_ndarray()


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
        argmax = F.np.argmax(x, axis=-1)
        match = F.where((F.npx.pick(x, argmax, axis=-1) >= self._threshold).as_nd_ndarray(), argmax.as_nd_ndarray(),
                        (F.np.ones_like(argmax) * -1).as_nd_ndarray()).as_np_ndarray()
        return match
