import cv2
from utils import img_util

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]


def load_dnn(prototxt, caffemodel):
    cv2.dnn_registerLayer('Crop', CropLayer)
    # Load the model.
    return cv2.dnn.readNet(prototxt, caffemodel)

def edge_detection(dnn, image,scalefactor=1.0, size=(500, 500), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False):
    inp = cv2.dnn.blobFromImage(image, scalefactor=scalefactor, size=size,
                            mean=mean,
                            swapRB=swapRB, crop=crop)
    dnn.setInput(inp)
    out = dnn.forward()
    out = cv2.resize(out[0, 0], (image.shape[1], image.shape[0]))
    out = (255 * out).astype("uint8")
    out = cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
    return out


def crop_image(dnn, image,scalefactor=1.0, size=(500, 500), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False):
    hed_image = edge_detection(dnn, image, scalefactor, size, mean, swapRB, crop)
    gray = cv2.cvtColor(hed_image, cv2.COLOR_BGR2GRAY)
    contours, last_cnt = img_util.findContour(gray, convert_binary=True)
    if (last_cnt is None):
        return None
    x,y,w,h = cv2.boundingRect(contours[len(contours)-1])
    #print('x {} y {} w {} h {}'.format(x,y,w,h))
    return image[y:y+h, x:x+w]
