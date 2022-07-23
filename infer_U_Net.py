from unet import Unet
from torchvision import transforms
import cv2
import torch as t
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt
"""Inference U net
In this model, including all the methods used for inferences U-Net, such as padding, conversion between numpy and tensor, etc.
"""

def padding(img):
    """Padding input image to square with out change the original ratio.
    square size == max length of the edge

    :param img: input image with shape [width, height, channel]
    :type img: numpy array
    :return: padded imagewith shape [max_edge, max_edge, channel]
    :rtype: numpy array
    """
    w,h,c = img.shape
    max_edge = np.max([w,h])
    frame_pad = np.zeros([max_edge,max_edge,c]).astype(np.int16)
    ext = np.abs(int((w-h)/2))

    if w < h:
        frame_pad[:,:,0] = np.pad(img[:,:,0],((ext,ext),(0,0)), 'constant', constant_values=(0,0))
        frame_pad[:,:,1] = np.pad(img[:,:,1],((ext,ext),(0,0)), 'constant', constant_values=(0,0))
        frame_pad[:,:,2] = np.pad(img[:,:,2],((ext,ext),(0,0)), 'constant', constant_values=(0,0))

    if w > h:
        frame_pad[:,:,0] = np.pad(img[:,:,0],((0,0),(ext,ext)), 'constant', constant_values=(0,0))
        frame_pad[:,:,1] = np.pad(img[:,:,1],((0,0),(ext,ext)), 'constant', constant_values=(0,0))
        frame_pad[:,:,2] = np.pad(img[:,:,2],((0,0),(ext,ext)), 'constant', constant_values=(0,0))
    return frame_pad

def unpadding(img, result):
    """Undo the padding, back to the original size, with help of the original image.

    :param img: original image, use the width and height information to undo the padding
    :type img: numpy array
    :param result: result from the U-Net, with shape [2, width, height]
    :type result: list[numpy array]
    :return: unpadded result
    :rtype: list[numpy array]
    """
    w,h,_ = img.shape
    max_edge = np.max([w,h])
    ext = np.abs(int((w-h)/2))

    resTemp = np.zeros([2,w,h])
    channel0 = cv2.resize(result[0], (max_edge,max_edge))
    channel1 = cv2.resize(result[1], (max_edge,max_edge))
    if w < h:
        resTemp[0,:,:] = channel0[ext:ext+w,:]
        resTemp[1,:,:] = channel1[ext:ext+w,:]

    if w > h:
        resTemp[0,:,:] = channel0[:,ext:ext+h]
        resTemp[1,:,:] = channel1[:,ext:ext+h]
    return resTemp


def numpy2tensor(img, compose):
    """cover numpy array to torch float tensor
    also cover the BGR color space to RGB

    :param img: input image
    :type img: numpy array
    :param compose: compose function which defined out of this function
    :type compose: torchvision.transform.Compose
    :return: image tensor
    :rtype: torch.floattensor
    """
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = padding(img)
    img = cv2.resize(img, (224,224))
    img = img / 255
    tensor = compose(img).type(t.FloatTensor)
    return t.unsqueeze(tensor,0)


def inference_NN(videos):
    """main function of inference U-Net

    :param videos: loaded video
    :type videos: list[numpy array]
    :return: U-Net result with same size as load video
    :rtype: numpy array
    """
    # path = "E:\\Cloud\\External storage\\Self_Gitlab\\TracO\\Training videos\\training{}.mp4".format(num)
    # print(path)
    compose = transforms.Compose([transforms.ToTensor()])
    net = Unet(n_channels=3, n_classes=2, recurrent=False, residual=True)
    ckp = t.load('ckp/checkpoint_{}.ckp'.format("Residual"), None)#'cuda')
    net.load_state_dict(ckp['state_dict'])

    results = []
    for frame in tqdm(videos):
        frame_tensor = numpy2tensor(frame,compose)#.cuda()
        res = net(frame_tensor).cpu().data.numpy()[0]
        res = unpadding(frame, res)
        results.append(res)

    return np.array(results)
