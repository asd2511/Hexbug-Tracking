
""" This module contains all the utillity function for load video and preprocessing
"""

import cv2
import numpy as np
from FrameObject import Frame

def visual(frame):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(frame[0])
    plt.figure()
    plt.imshow(frame[1])
    plt.show()

def normalize(frame, thres):
    """normalize the image

    :param frame: frame image
    :type frame: numpy array
    :param thres: threshold for the binary mask
    :type thres: float
    :return: normalized image
    :rtype: numpy array
    """
    frame[0] = (frame[0] - np.min(frame[0]))
    frame[0] /= np.max(frame[0])
    frame[1] = (frame[1] - np.min(frame[1]))
    frame[1] /= np.max(frame[1])
    frame[frame<thres] = 0
    frame[frame>=thres] = 1
    frame = frame.astype(np.uint8)
    return frame

def erodeUtilFit(img, num:int, maxIter=10, sizeScale=10):
    """do erode for the mask, make the component close the the number of bugs

    :param img: mask image
    :type img: numpy array
    :param num: number of bugs
    :type num: int
    :param maxIter: max number of iteration, defaults to 5
    :type maxIter: int
    :param sizeScale: scale size of the erode, defaults to 2
    :type sizeScale: int
    :return: component
    :rtype: tuple()
    """
    for i in range(maxIter):
        kernel = np.ones((sizeScale*i+1,sizeScale*i+1),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = 1)
        output = cv2.connectedComponentsWithStats(erosion, 8, cv2.CV_32S)
        if output[0]-1 <= num:
            return output, erosion
    return output, erosion

def connectComponentForVideos(videos, thres: float = 0.70):
    """ compute connect component for video

    :param videos: load video
    :type videos: numpy array
    :param thres: threshold for get the binary mask, defaults to 0.75
    :type thres: float
    """

    num = []
    res = []
    frames = []
    for frame in videos:
        frame = normalize(frame,thres)
        frames.append(frame[0])
        # visual(frame)
        output0 = cv2.connectedComponentsWithStats(frame[0], 8, cv2.CV_32S)

        num.append(output0[0])
        res.append([output0,frame[1]])
    counts = np.bincount(np.array(num))
    numBug = np.argmax(counts)-1
    print(numBug)
    for i in range(len(videos)):
        res[i][0], frames[i] = erodeUtilFit(frames[i], numBug, maxIter=5, sizeScale=2)
    return np.array(res)

def getVideoFromPath(path):
    """Load video from given path

    :param path: path to the video end with ".mp4"
    :type path: string
    :return: list of frame, shape = [#frame, width, height]
    :rtype: list[frame]
    """
    cap = cv2.VideoCapture(path)
    video = []
    while 1:
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    return video

def component2trackObject(datas,videos):
    """cover component to tracking object

    :param datas: compute result from cv2.connectedComponentsWithStats
    :type datas: tuple
    :param videos: load video
    :type videos: numpy array
    :return: list of frame object
    :rtype: list[Frame]
    """
    video = []
    for data,img in zip(datas,videos):

        # from infer_U_Net import padding
        # img = padding(img)
        # img = cv2.resize(img, (224,224))

        frame = Frame(data, img)
        video.append(frame)
    return video
