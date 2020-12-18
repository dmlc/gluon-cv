"""

"""

class TrackState:
    """
    States of the track.
    The track follows the simple state machine as below:

    Active: time_since_update always set to 1
        1. If confidence < keep_alive_threshold, goto Missing
        2. If the track is suppressed in track NMS, goto Missing
    Missing: every timestep the missing track increment time_since_update by one
        1. If the track is updated again, goto Active
        2. If time_since_update > max_missing, goto Deleted
    Deleted: This is an absorbing state
    """
    Active = 1
    Missing = 2
    Deleted = 3


class Track:
    """
    This class represents a track/tracklet used in the FAR Tracker
    It has the following properties

    *******************************************************
    mean: 4-tuple representing the (x0, y0, x1, y1) as the current state (location) of the tracked object
    track_id: the numerical id of the track
    age: the number of timesteps since its first occurrence
    time_since_update: number of time-steps since the last update of the its location
    state: the state of the track, can be one in `TrackState`
    confidence_score: tracking_confidence at the current timestep

    source: a tuple of (anchor_indices, anchor_weights)
    attributes: np.ndarray of additional attributes of the object
    *******************************************************

    It also has these configs
    keep_alive_thresh: the minimal tracking/detection confidence to keep the track in Active state
    max_missing: the maximal timesteps we will keep searching for this track when missing before we mark it as deleted
    *******************************************************

    """

    def __init__(self, mean, track_id, source, keep_alive_thresh=0.1, max_missing=30,
                 attributes=None, class_id=0, linked_id=None):
        self.mean = mean
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Active
        self.confidence_score = 1.

        self.matched = False
        self.match_score = 999
        self.current_detection_index = None
        self.max_missing = max_missing

        self.next_frame_anchor_id = -1
        self.keep_alive_thresh = keep_alive_thresh
        self.attributes = attributes
        self.source = source
        self.class_id = class_id
        self.linked_id = linked_id

    @property
    def display_id(self):
        if self.linked_id is not None:
            return self.linked_id
        else:
            return self.track_id

    @property
    def linkable(self):
        return self.linked_id is None

    def link_to(self, track):
        self.linked_id = track.display_id

    def predict(self, motion_model=None):
        """
        Parameters
        ----------
        motion_model : if not None, predict the motion of this track given its history
        """
        if motion_model:
            self.mean, self.covariance = motion_model(self.mean, self.covariance)

        self.age += 1
        self.time_since_update += 1

        # If the track has long been missing, mark it as deleted
        if self.time_since_update > self.max_missing:
            self.state = TrackState.Deleted

    def update(self, bbx, source=None, attributes=None):
        """
        Update the state of the track. We override the predicted track position.
        Updating the track will keep or flip its state as Active
        If the confidence of detection is below the keep_alive_threshold, we will mark this track as missed.
        ----------
        bbx : new detection location of this object
        attributes: some useful attributes of this object at this frame, e.g. landmarks
        """
        self.mean = bbx[:4]
        self.confidence_score = bbx[4]
        self.time_since_update = 0
        self.state = TrackState.Active
        self.attributes = attributes
        self.source = source

        if self.confidence_score < self.keep_alive_thresh:
            self.mark_missed()

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        # only an active track can be marked as missed
        if self.state == TrackState.Active:
            self.state = TrackState.Missing

    def is_mising(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Missing

    def is_active(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Active

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
