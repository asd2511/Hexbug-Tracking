
from track_util import *
from Locator import Locator
from save_util import *
from Tracker import Tracker
from infer_U_Net import inference_NN
import warnings
warnings.filterwarnings("ignore", category=Warning)

saveNN = False
saveVideo = True
showVisual = True
saveTraco = False

def run(path, name):
    """Main function for the program

    :param path: the path of the video that we want to analysis
    :type path: string
    """
    videos = getVideoFromPath(path)
    result = inference_NN(videos)

    if saveNN:
        saveNNresult(path, result)
    data = connectComponentForVideos(result)
    data = component2trackObject(data,videos)
    tracker = Tracker(data)
    tracker.run()
    headLocator = Locator(tracker.data)
    headLocator.locateWithMask()
    headLocator.locateWithMotionVec()
    vis = Visualizer(tracker.data, name=name, saveVideo=saveVideo)
    if showVisual:
        vis.play(showHead=True, showBBox=True)
    if saveTraco:
        final = save2traco(tracker.data)
        saveTracoResult(final, path)

if __name__ == "__main__":
    # name = "8"
    # path = "Training videos\\training0{}.mp4".format(name)
    for i in range(1,99):
        name = str(i)
        path = "Training videos\\training0{}.mp4".format(name)
        # path = "testVideo\\test000{}.mp4".format(name)
        run(path, name)


    # path = "testVideo\\test000{}.mp4".format(1)
    # run(path, str(5))
