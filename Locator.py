
import cv2
import numpy as np
from sklearn.cluster import KMeans
class Locator:
    """locate head position base of mask
    """
    def __init__(self, data) -> None:
        """init function

        :param data: list of frameObject
        :type data: list[Frame]
        """
        self.data = data

    def locateWithMask(self):
        """locate head position with mask
        If successful, get the head position, otherwise provide candidates
        """
        for frame in self.data:
            mask = frame.headMask
            for bug in frame.hexbugs:
                bug.headCandidate = self.computeHead(bug.label,mask)
                if len(bug.headCandidate) == 1:
                    bug.headPos = bug.headCandidate[0]

    def locateWithMotionVec(self):
        """locate head position with motion vector
        Select head position from candidates based on motion vector
        """
        for frame in self.data:
            for bug in frame.hexbugs:
                if bug.headPos is not None: continue
                newPos = bug.center + bug.motionVector
                candidate = bug.headCandidate
                distance = []
                for x,y in candidate:
                    dis = np.sqrt((newPos[0]-x)**2+(newPos[1]-y)**2)
                    distance.append(dis)
                bug.headPos = candidate[np.argmin(distance)][::-1]

    def computeHead(self, label, maskOfAll):
        """compute head position
        compute the component of the head mask.
        if we only get one position, then head position it is.
        else use K-Means compute head position's candidate

        :param label: hexbug mask
        :type label: numpy array
        :param maskOfAll: head mask for all bugs in one frame (output from u-net)
        :type maskOfAll: numpy array
        :return: head position's candidate
        :rtype: list[]
        """
        mask = (label * maskOfAll).astype(np.uint8)
        num,_,_,centers = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        if num == 2:
            return centers[1:]
        else:
            # no head or muti head: use kmeans get 2 candidate
            return self.getHeadFromKmean(label)

    def getHeadFromKmean(self, label):
        """K-Means compute the head position's candidate
        Process the mask using K-Means algorithm with k=2. Always get two head position candidates
        :param label: mask fo hexbug
        :type label: numpy array
        :return: candidate
        :rtype: list[]
        """
        kmeans = KMeans(n_clusters=2)
        if np.argwhere(label==1).shape[0] == 1:
            return np.argwhere(label==1)
        kmeans.fit(np.argwhere(label==1))
        return kmeans.cluster_centers_
