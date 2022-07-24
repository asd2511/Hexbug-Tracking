from FrameObject import HexObject
import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
class Segment:
    """a segment will save all the hexbug which recognize as same hexbug
    """
    def __init__(self, id:int) -> None:
        """init function

        :param id: id of current hexbug, also is the id of current segment
        :type id: int
        """
        self.id = id
        self.sequence = []
    def push(self,data:HexObject):
        """push new hexbug into segment
        compute the motionVector when push new bug.
        Motion vector = current position [center] - previous position [center]
        :param data: new hexbug object
        :type data: HexObject
        """
        self.sequence.append(data)

        currentPos = np.array(self.sequence[-1].center)
        prevPos = np.array(self.sequence[-2].center) if len(self.sequence) != 1 else np.array([0,0])

        self.getLast().motionVector = currentPos - prevPos
    def getLast(self):
        """return the last hexbug

        :return: last hexbug
        :rtype: HexObject
        """
        return self.sequence[-1]

class Tracker:
    """main tracking method
    """
    def __init__(self, videos, colorDist:int = 7) -> None:
        """init function

        :param videos: list of Frame object
        :type videos: list[Frame]
        :param colorDist: min color distance, used to calculate color similarity , defaults to 7
        :type colorDist: int
        """
        self.colorDist = colorDist
        self.length = len(videos)
        self.data = videos
        self.currentID = 0
        self.segments = {}

    def run(self):
        """main function
        1. loop for the list of Frame, get all hexbugs of two adjacent frames.
        2. Through the Hungarian algorithm, use IOU, BBox, L2 to calculate the correlation of all hexbugs.
        3. Use majority voting to get the final match.
        4. Assign the corresponding id to the matched result.
        5. For hexbugs that do not find a match, find the closest color similarity in all segments as the matching target.
        6. If no matching id is found, set to new id
        """
        for i in range(self.length-1):
            current = self.data[i].hexbugs
            next = self.data[i+1].hexbugs
            if i == 0:
                self.initFirstFrame()
            # do matching and major Vote
            res_IOU  = hungarian(self.computeCostMatrix(current,next, func=self.computeIOU))
            res_BBox = hungarian(self.computeCostMatrix(current,next, func=self.computeBBOX))
            res_L2   = hungarian(self.computeCostMatrix(current,next, func=self.computeL2))
            prev_idx, next_idx = self.majorVote([res_IOU,res_BBox,res_L2])
            # assign matched result
            for idx_prev, idx_next in zip(prev_idx, next_idx):
                newID = current[idx_prev].id
                next[idx_next].id = newID
                self.pushNew(next[idx_next], newID)
            for bug in next:
                if bug.id == -1:
                    # color match
                    for k in self.segments.keys():
                        if self.segments[k].getLast().mainHSV[0] - bug.mainHSV[0] <=self.colorDist:
                            bug.id = k
                            self.pushNew(bug,k)
                            break
                # set new id
                if bug.id == -1:
                    self.pushNew(bug,self.currentID)
        print(self.currentID)

    def initFirstFrame(self):
        """init segment in first frame
        every hexbug have a new id
        """
        temp = self.data[0].hexbugs
        for i in range(len(temp)):
            self.pushNew(temp[i], i)

    def pushNew(self, bug, id):
        """push new hexbug to Segment by id. if id not exist, create new segment


        :param bug: new Hexbug object
        :type bug: HexObject
        :param id: id of the hexbug
        :type id: int
        """
        bug.id = id
        if id not in self.segments.keys():
            self.segments[id] = Segment(id)
            self.currentID += 1
        self.segments[id].push(bug)

    def computeIOU(self, a,b):
        """compute the Intersection over Union base on mask

        :param a: hexbug A
        :type a: HexObject
        :param b: hexbug B
        :type b: HexObject
        :return: mask iou of bug a and bug b
        :rtype: float
        """
        a = a.label
        b = b.label
        upper = a * b
        lower = a + b
        lower[lower!=0] = 1
        return np.sum(upper)/np.sum(lower)

    def computeBBOX(self, a,b):
        """compute the Intersection over Union base on bbox

        :param a: hexbug A
        :type a: HexObject
        :param b: hexbug B
        :type b: HexObject
        :return: bbox iou of bug a and bug b
        :rtype: float
        """
        a = a.bbox
        b = b.bbox
        area_a = (a[3]-a[1])*(a[2]-a[0])
        area_b = (b[3]-b[1])*(b[2]-b[0])
        w = min(a[2],b[2]) - max(a[0],b[0])
        h = min(a[3],b[3]) - max(a[1],b[1])
        if w <= 0 or h <= 0:
            return 0
        area_c = w * h
        return area_c / (area_a + area_b - area_c)

    def computeL2(self, a,b):
        """compute the distance of bugs

        :param a: hexbug A
        :type a: HexObject
        :param b: hexbug B
        :type b: HexObject
        :return: L2 distance between bug a and bug b
        :rtype: float
        """
        a = a.center
        b = b.center
        return -np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    def computeCostMatrix(self,  prev, next, func=computeIOU):
        """Compute the cost matrix, since the method linear_sum_assignment (aka hungarian)
        can only calculate the minimum distance, take a negative value for the calculated result.

        :param prev: prev hexbug
        :type prev: HexObject
        :param next: next hexbug
        :type next: HexObject
        :param func: cost function, defaults to computeIOU
        :type func: function
        :return: cost matrix
        :rtype: numpy array
        """
        m,n = len(prev), len(next)
        costMatrix = np.zeros([m,n])
        for i in range(m):
            for j in range(n):
                costMatrix[i,j] = func(prev[i],next[j])

        return -1*costMatrix

    def majorVote(self, data):
        """majority voting method

        :param data: results compute by hungarian method
        :type data: list[np.array]
        :return: match index of prev and next
        :rtype: tuple(list, list)
        """
        prev = []
        next = []
        res_prev = []
        res_next = []
        for _prev,_next in data:
            prev.append(_prev)
            next.append(_next)
        prev,next = np.array(prev),np.array(next)
        matchNum = prev.shape[1]
        for i in range(matchNum):
            counts_prev = np.bincount(prev[:,i])
            res_prev.append(np.argmax(counts_prev))
            counts_next = np.bincount(next[:,i])
            res_next.append(np.argmax(counts_next))
        return res_prev, res_next
