"""All methods related to saving data and visualizing data are stored in this model.
"""

import cv2
import json
import numpy as np

saveFolderVideo = "result_mp4"
saveFolderNN = "result_npy"
saveFolderTraco = "result_traco"

def save2traco(data):
    """save function, save tracking result to traco type

    :param data: tracking data
    :type data: list[Frame]
    :return: tracking result with format of Traco
    :rtype: dict
    """
    rois = []
    for i in range(len(data)):
        frame = data[i]
        for bug in frame.hexbugs:
            roi = bug.getROI(i)
            rois.append(roi)
    return {"rois": rois}

def saveNNresult(path,results):
    """save the U-Net result

    :param path: Save path
    :type path: string
    :param results: U-Net result
    :type results: numpy array
    """
    saveName = path.split("\\")[-1].replace(".mp4",".npy")
    with open(saveFolderNN + "/" +saveName, "wb") as f:
        np.save(f, np.array(results))

def saveTracoResult(res, path):
    """save traco result into .traco file

    :param res: save path
    :type res: string
    :param path: tracking result
    :type path: list[Frame]
    """
    saveName = path.split("\\")[-1].replace(".mp4",".traco")
    with open(saveFolderTraco + "/"+saveName, 'w') as outfile:
        json.dump(res,  outfile, sort_keys=True, indent=4,)

class Visualizer:
    """Visualization of Operation Results
    """
    def __init__(self,frameObject, name, saveVideo) -> None:
        """init function

        :param frameObject: frame object
        :type frameObject: Frame
        :param name: idx of current video
        :type name: str(int)
        :param saveVideo: flag if we save the video
        :type saveVideo: boolean
        """
        self.frameObject = frameObject
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # encoder
        m,n,c = frameObject[0].image.shape
        self.out = cv2.VideoWriter(saveFolderVideo + '/result_{}.mp4'.format(name), fourcc, 10.0, (n, m))  # writer
        # self.out = cv2.VideoWriter(saveFolderVideo + '/result_{}.mp4'.format(name), fourcc, 10.0, (224, 224))  # writer
        self.saveVideo = saveVideo

    def play(self, showHead=False, showBBox=False):
        """play the video

        :param showHead: flag if show the head position, defaults to False
        :type showHead: bool
        :param showBBox: flag if show the BBox, defaults to False
        :type showBBox: bool
        """
        for frame in self.frameObject:
            img = frame.image
            if showHead:
                img = self.drawHead(img,frame.hexbugs)
            if showBBox:
                img = self.drawBBox(img,frame.hexbugs)
            img = img.astype(np.uint8)
            if self.saveVideo:
                self.out.write(img)
            cv2.imshow("HexBug training Visual - Group SK", img)
            cv2.waitKey(100)
    def drawHead(self,img,hexbugs):
        """draw head positions

        :param img: target image
        :type img: numpy array
        :param hexbugs: target hexbugs
        :type hexbugs: list[HexObject]
        :return: draw image
        :rtype: numpy array
        """
        for bug in hexbugs:
            pos = bug.headPos
            img = cv2.circle(img, (int(pos[0]),int(pos[1])), 10, (0,0,255), -1)
        return img
    def drawBBox(self,img,hexbugs):
        """draw bbox

        :param img: target image
        :type img: numpy array
        :param hexbugs: target hexbugs
        :type hexbugs: list[HexObject]
        :return: draw image
        :rtype: numpy array
        """
        for bug in hexbugs:
            pos = bug.bbox
            img = cv2.rectangle(img, (int(pos[0]),int(pos[1])),(int(pos[0])+int(pos[2]),int(pos[1])+int(pos[3])), (0,255,0), 2)
        return img


