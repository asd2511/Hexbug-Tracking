
import cv2
import numpy as np
class Frame:
    """FrameObject 
    Save all the information that should be contained in a frame object.
    """
    def __init__(self, data, img):
        """init function save data in frame Object

        :param data: result from U-Net
        :type data: numpy array shape [2,width, height]
        :param img: frame image
        :type img: numpy array shape [width, height, 3]
        """
        self.image = img
        self.headMask = data[1]
        num, labels, bboxes, centers = data[0]
        self.hexbugs = []

        for i in range(1,num):
            label = labels==i
            bbox = bboxes[i]
            center = centers[i]
            self.hexbugs.append(HexObject(label, bbox, center, self.computeMainColor(label)))

    def computeMainColor(self, label):
        """Compute a Main color of current bug
        cover the color into HSV color space reduce the impact of Saturation, Value

        :param label: mask image of current bug
        :type label: numpy array
        :return: color
        :rtype: int
        """
        mask = np.zeros_like(self.image)
        mask[:,:,0] = label
        mask[:,:,1] = label
        mask[:,:,2] = label
        img = cv2.cvtColor(self.image.astype(np.uint8),cv2.COLOR_BGR2HSV)
        c = img[label]
        unique, counts = np.unique(c, axis=0, return_counts=True) # compute mode
        return unique[np.argmax(counts)]

class HexObject:
    """Hexbug Object
    Save all the information that should be contained in a hexbug object
    """
    def __init__(self, label, bbox, center, color) -> None:
        """init function

        :param label: mask image of one bug
        :type label: numpy array
        :param bbox: bounding box
        :type bbox: list[x,y,w,h]
        :param center: center position of bbox = [x+0.5w,y+0.5h]
        :type center: list[x,y]
        :param color: main color is hsv [hue]
        :type color: int
        """
        self.label = label
        self.bbox = bbox
        self.center = center
        self.mainHSV = color

        self.motionVector = None

        self.headPos = None
        self.headCandidate = []
        self.id = -1

    def getROI(self, frameNum:int):
        """generate the return dict follow the rule of traco

        :param frameNum: current frame id
        :type frameNum: int
        :return: a dict which save all the information we need to print out into traco file
        :rtype: dict
        """
        return {
                    "z": frameNum,
                    "id": self.id,
                    "pos": self.headPos.tolist()
                }


