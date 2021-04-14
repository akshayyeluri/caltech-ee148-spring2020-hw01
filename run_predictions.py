import os
import cv2
import numpy as np
import json
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# set the path to the downloaded data: 
data_path = '../RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = './hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

with open('filter.pkl', 'rb') as f:
    filt = pickle.load(f)
    
def read_im(fname, data_path=data_path):
    # read image using PIL:
    I = Image.open(os.path.join(data_path,fname))
    # convert to numpy array:
    return np.asarray(I)
    
    
def get_circles(file, filepath=data_path):
    '''
    This function returns a list of centers and radii
    for red circles in the image given by file
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(os.path.join(filepath,file))
    cimg = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ranges we'll consider red
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskr = cv2.add(mask1, mask2)

    m,n,_ = img.shape

    # hough circle detect
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                               param1=50, param2=10, minRadius=0, maxRadius=30)

    
    # needs to be in top bound proportion of image, since lights are high
    # usually
    bound = 4.0 / 10 
    
    centers = []
    radii = []
    
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for (x,y,r) in r_circles[0, :]:
            # Ensure circle we're finding is in the picture
            if x > n or y > m or y > m*bound: 
                continue
            centers.append((x,y))
            radii.append(r)
            
    return centers, radii



def detect_red_light(fname, filt=filt, thresh = 0.85, filepath=data_path):
    '''
    This function takes a filename <fname> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    bboxes = []
    centers, radii = get_circles(fname, filepath)
    I = read_im(fname, filepath)
    m, n, _ = I.shape
    
    for (x,y), r in zip(centers, radii):
        # [1, 2] is aspect ratio, 3 is scaling since width of
        # traffic light is about 3 times the radius of the
        # actual red circle
        dims = np.array([1, 2]) * r * 3 
        anchor = (x - (dims[0] // 2), y - dims[1] // 4)
        bbox = [anchor[0], anchor[1], anchor[0] + dims[0], anchor[1] + dims[1]]
        bbox = [max(0, bbox[0]), max(0, bbox[1]),
                min(n, bbox[2]), min(m, bbox[3])]
        
        I_spliced = I[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        pil = Image.fromarray(I_spliced)
        I2 = np.asarray(pil.resize((filt.shape[1], filt.shape[0])))
        I3 = I2 / np.linalg.norm(I2)
        if (np.sum(I3 * filt) > thresh):
            bboxes.append(bbox)
                
    for i in range(len(bboxes)):
        assert len(bboxes[i]) == 4
    
    return bboxes



preds = {}
for i in range(len(file_names)):
    preds[file_names[i]] = [[int(n) for n in val] \
                            for val in detect_red_light(file_names[i])]
    

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
