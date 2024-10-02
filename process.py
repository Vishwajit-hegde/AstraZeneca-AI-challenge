
""" 
Contains the functions and classes to extract plots from image
"""

import os
import cv2
import imutils
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract as tess
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


# HELPER FUNCTIONS

def path(root, file):
    ''' Returns full path to a file within root '''
    return os.path.join(root, str(file)+'.png')


def show_img(img, gray=False):
    ''' Plots an image '''
    plt.figure(figsize=(10, 8))
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    
def remove_jerks(x):
    ''' Smoothens any abrupt changes in the sequence '''
    for i in range(len(x)-3):
        c1 = abs(x[i+1, 1] - x[i, 1])/min(x[i, 1], x[i+1, 1]) > 1.0
        c2 = abs(x[i+1, 1] - x[i+2, 1])/min(x[i+1, 1], x[i+2, 1]) > 1.0
        c3 = abs(x[i+2, 1] - x[i+3, 1])/min(x[i+2, 1], x[i+3, 1]) > 1.0
        if c1 & (c2 | c3):
            x[i+1, 1] = x[i, 1]
    return x

    
def length(line, along='x'):
    ''' Finds length of a line along an axis '''
    (x1, y1), (x2, y2) = line
    if along == 'x':
        return np.abs(x2-x1)
    elif along == 'y':
        return np.abs(y2-y1)
    else:
        raise ValueError('along value is wrong')
    
    
def get_neighbors(arr, loc, radius=2):
    ''' Finds pixel values of neighbors in a 2*r+1 sided box around the target pixel '''
    x, y = loc
    y_lower, y_upper = max(0, y-radius), min(arr.shape[0], y+radius+1)
    x_lower, x_upper = max(0, x-radius), min(arr.shape[1], x+radius+1)
    if len(arr.shape) == 3:
        return arr[y_lower:y_upper, x_lower:x_upper, :]
    elif len(arr.shape) == 2:
        return arr[y_lower:y_upper, x_lower:x_upper]
    
    
def percent_change(bs1, bs2):
    ''' Percent change in y-axis values of two arrays '''
    return abs(bs1[1]-bs2[1])/min(bs1[1], bs2[1])
    

class Extractor:
    ''' 
    This class is responsible for extracting the plot region from the input image.
    This will help us to ignore information outside such as legend or other text.

    Args:
        img <numpy.ndarray> : Image in the form of a numpy array. Preffered dtype is np.uint8

    Returns:
        If you use the .run() method:
            cropped plot region, binary thresholded cropped plot region, edge detected cropped plot region

        If you use the .run_for_text() method (BEING USED BY THE CODE):
            cropped plot region, cropped plot region with axes included, x-correction value, y-correction value

            Note: The correction values are the amount of pixels added to the coordinates due to inclusion
            of the plot axes. 
    '''

    def __init__(self, img):
        self.img = img
        self.thresh = None
        self.edged = None
        self.plot_bounds = None
        self.cropped_img_full = None
        self.cropped_img_thresh = None
        self.cropped_img_edged = None

    def preprocess_img_(self, img):
        ''' Grayscaling and thresholding '''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        return thresh

    def find_edges_(self, img):
        ''' Edge detection '''
        edged = cv2.Canny(img, 50, 100)
        return edged

    def length(self, line, along='x'):
        ''' Helper function to find length of a line along an axis '''
        (x1, y1), (x2, y2) = line
        if along == 'x':
            return np.abs(x2-x1)
        elif along == 'y':
            return np.abs(y2-y1)
        else:
            raise ValueError('along value is wrong')
    
    def get_bounds_(self, img):
        ''' Detects plot axes and gets coordinates of plot bounds '''
        line_coords = []
        lines = cv2.HoughLinesP(img, 1, np.pi/180, 100, 50, 2)
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            line_coords.append([(x1, y1), (x2, y2)])

        horz_longest = lines[np.argmax([self.length(l, 'x') for l in line_coords])]
        vert_longest = lines[np.argmax([self.length(l, 'y') for l in line_coords])]
        blines = np.array([horz_longest, vert_longest]).reshape(-1, 4)
        xmin, xmax = blines[:, [0, 2]].min(), blines[:, [0, 2]].max()
        ymin, ymax = blines[:, [1, 3]].min(), blines[:, [1, 3]].max()
        self.plot_bounds = [xmin, xmax, ymin, ymax]

    def run(self):
        ''' Check main docstring for .run() method '''
        self.thresh = self.preprocess_img_(self.img)
        self.edged = self.find_edges_(self.thresh)
        self.get_bounds_(self.edged)
        x1, x2, y1, y2 = self.plot_bounds
        x1, x2 = x1+5, x2-5
        y1, y2 = y1+5, y2-5
        self.cropped_img_full = self.img[y1:y2, x1:x2]
        self.cropped_img_thresh = self.thresh[y1:y2, x1:x2]
        self.cropped_img_edged = self.edged[y1:y2, x1:x2]
        return self.cropped_img_full, self.cropped_img_thresh, self.cropped_img_edged
    
    def run_for_text(self):
        ''' Check main docstring for .run_for_text() method '''
        self.thresh = self.preprocess_img_(self.img)
        self.edged = self.find_edges_(self.thresh)
        self.get_bounds_(self.edged)
        x1, x2, y1, y2 = self.plot_bounds
        x1, x2 = x1+5, x2-5
        y1, y2 = y1+5, y2-5
        self.cropped_img_full = self.img[y1:y2, x1:x2]
        self.cropped_img_thresh = self.thresh[y1:y2, x1:x2]
        self.cropped_img_edged = self.edged[y1:y2, x1:x2]
        h, w, c = self.cropped_img_full.shape
        x1_ = max(0, x1 - int(0.2 * w))
        y2_ = min(y2 + int(0.2 * h), self.img.shape[0])
        cropped_with_axes = self.img[y1:y2_, x1_:x2]
        x_corr = int(0.2 * w)
        y_corr = int(0.2 * h)
        return self.cropped_img_full, cropped_with_axes, x_corr, y_corr
    

class DeconstructGraph:
    ''' 
    Main class that performs all the operations necessary to get the output.
    '''
    
    def __init__(self, img, img_name):
        self.img_name = img_name
        self.ext = Extractor(img)
        
    def find_textboxes(self, cropped):
        ''' Returns the bounding boxes of detected text objects in image '''

        # Morphological transforms to extract text regions
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask = np.zeros(bw.shape, dtype=np.uint8)
        bbox_cands = []

        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            bbox_cands.append([x, y, w, h])

        # Choose only those whose area wrt image area is less than 10%
        img_area = cropped.shape[0] * cropped.shape[1]
        bboxes = []
        for b in bbox_cands:
            if (b[2] * b[3])/img_area < 0.1:
                bboxes.append(b)
        return bboxes

    def extract_series(self, cropped, neighbor_radius=2):
        # Remove any text within the image
        bboxes = self.find_textboxes(cropped)
        for b in bboxes:
            x, y, w, h = b
            cropped[y:y+h, x:x+w] = 255

        # Grayscale and find all the corner points
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 0, 0.01, 10).reshape(-1, 2).astype('int')
        results = {}
        all_selected = []

        # Extract colors of the points and some processing required for further steps
        for i, c in enumerate(corners):
            neighs = get_neighbors(cropped, c, neighbor_radius).reshape(-1, 3)
            selected = neighs[np.argmax(neighs.std(axis=1))]
            results[i] = selected.tolist()
            if selected.tolist() not in all_selected:
                all_selected.append(selected.tolist())

        # Clustering to find which point belongs to which sequence
        all_colors = np.array(list(results.values()))
        db = DBSCAN(eps=30).fit(all_colors)
        labels = db.labels_
        final_data = {}

        # Post processing and accumulation
        for l in np.unique(labels):
            if l != -1:
                locs = np.where(labels == l)[0]
                data = corners[locs]
                data[:, 1] = cropped.shape[0] - data[:, 1]
                final_data[l] = data

        for i, v in final_data.items():
            v = v[np.argsort(v[:, 0])]
            final_data[i] = remove_jerks(v)
        return final_data

    def get_text_data(self, cropped, withaxes):
        ''' 
        Finds bounding boxes around text objects in image and returns the values
        of whichever boxes could be decoded to numbers by Tessearct
        '''
        # Take image with axes and the cropped image. In the image with axes, set the 
        # region occupied by the cropped image to white. Now you only have the axes
        withaxes[:cropped.shape[0]+5, -(cropped.shape[1]+5):] = 255
        withaxes = cv2.cvtColor(withaxes, cv2.COLOR_BGR2GRAY)
        _, withaxes = cv2.threshold(withaxes, 0, 255, cv2.THRESH_OTSU)
        withaxes = np.repeat(np.expand_dims(withaxes, axis=-1), 3, axis=-1)
        bboxes = self.find_textboxes(withaxes)

        # Convenient holder for all text bounding boxes
        boxes = []
        for i in range(len(bboxes)):
            x, y, w, h = bboxes[i]
            center = [x+w//2, y+h//2]
            crop = withaxes[y:y+h, x:x+w]
            boxes.append({'box': [x, y, w, h], 'center': center, 'crop': Image.fromarray(crop)})

        # Separate bounding boxes into those belonging to x-axis and y-axis
        xboxes, yboxes = [], []
        xboxes.append(boxes[0]['center'])

        for i in range(1, len(boxes)):
            if percent_change(boxes[i]['center'], boxes[i-1]['center']) < 0.1:
                xboxes.append(boxes[i]['center'])
            else:
                yboxes.extend([boxes[j]['center'] for j in range(i, len(boxes))])
                break

        # Extract text from bboxes using Tesseract. Whenever bad/no text is detected, puts None
        text = []
        for i in range(len(boxes)):
            st = tess.image_to_string(boxes[i]['crop'], lang='eng', config='--psm 7')
            if '\n' in st:
                st = st.split('\n')[0]
                try:
                    text.append(float(st))
                except:
                    text.append('None')
            else:
                text.append('None')

        for i, b in enumerate(boxes):
            b.update({'axis': 'x' if b['center'] in xboxes else 'y', 'text': text[i]})
        return boxes

    def find_points(self, withaxes, boxes, series, x_corr, y_corr):
        ''' Uses whatever data was generated earlier to find final point coordinates '''

        x_points, y_points = [], []
        for i in range(len(boxes)):
            if boxes[i]['axis'] == 'x':
                x_points.append((boxes[i]['center'], boxes[i]['text']))
            else:
                y_points.append((boxes[i]['center'], boxes[i]['text']))

        # Sort non-null x and y axis values in ascending order of coordinates
        x_ranks = np.argsort([s[0][0] for s in x_points])
        y_ranks = np.argsort([s[0][1] for s in y_points])
        x_points = [x_points[i] for i in x_ranks if x_points[i][1] != "None"]
        y_points = [y_points[i] for i in y_ranks if y_points[i][1] != "None"]
        
        # Switch y-axis direction from top-down to bottom-up
        for i in range(len(y_points)):
            y_points[i][0][1] = withaxes.shape[0] - y_points[i][0][1]

        # Point correction - remove useless or bad recorded points
        # (Can't vouch for the correctly recorded points tho)
        x_flagged, y_flagged = [], []
        for i in range(len(x_points)-1):
            if x_points[i+1][1] < x_points[i][1]:
                x_flagged.append(x_points[i+1])

        for i in range(len(y_points)-1):
            if y_points[i+1][1] > y_points[i][1]:
                y_flagged.append(y_points[i+1])

        x_points = [x for x in x_points if x not in x_flagged]
        y_points = [x for x in y_points if x not in y_flagged]
        resize_x, resize_y = False, False                                  # If axes cannot be captured correctly, set to True
        xv1, xv2, yv1, yv2 = 0, 0, 0, 0
        unsure = False                                                     # If axes cannot be captured correctly, set to True

        # Find anchor points on each axis for interpolation
        try:
            i = 0
            while (xv1 >= xv2) and (i < len(x_points)-1):
                if i > 0:
                    unsure = True
                x1, xv1, x2, xv2 = x_points[i][0][0], x_points[i][1], x_points[i+1][0][0], x_points[i+1][1]
                i += 1

            if i >= len(x_points)-1:
                resize_x = True
        except:
            resize_x = True

        try:
            i = 0
            while (yv1 >= yv2) and (i < len(y_points)-1):
                if i > 0:
                    unsure = True
                y2, yv2, y1, yv1 = y_points[i][0][1], y_points[i][1], y_points[i+1][0][1], y_points[i+1][1]
                i += 1

            if i >= len(y_points)-1:
                resize_y = True
        except:
            resize_y = True

        # Generate interpolation rules with values found above
        xmin, xmax, ymin, ymax = self.ext.plot_bounds
        if not resize_x:
            x_rule = lambda x: xv1 + (xv2 - xv1)*float((x + x_corr - x1)/(x2 - x1))
        if not resize_y:
            y_rule = lambda y: yv1 + (yv2 - yv1)*float((y + y_corr - y1)/(y2 - y1))

        # Interpolate series for new values using rules (if resize_axis is not True)
        new_series = {}
        for k, s in series.items():
            s = np.asarray(s).reshape(-1, 2).astype('float')
            for i in range(len(s)):
                if not resize_x:
                    s[i, 0] = x_rule(s[i, 0])
                if not resize_y:
                    s[i, 1] = y_rule(s[i, 1])
            new_series[k] = s

        warnings = []

        # Warnings
        if unsure:
            warnings.append(f"[WARN] {self.img_name} - Scales on some axes haven't been captured correctly. The shown scale could be inaccurate.")

        if resize_x or resize_y:
            warnings.append(f"[WARN] {self.img_name} - Some axes could not be calibrated to required scale due to OCR failure. Scaling those to lie between 0 and 1, please calibrate manually.")

        # If the algo cannot capture some axis scale correctly, it resizes the values to lie between 0 and 1 so user can calibrate manually 
        x_vals, y_vals = [], []
        for s in series.keys():
            x_vals.extend(np.array(series[s])[:, 0].tolist())
            y_vals.extend(np.array(series[s])[:, 1].tolist())
        x_vals, y_vals = np.array(x_vals), np.array(y_vals)
        xmin, xmax, ymin, ymax = float(x_vals.min()), float(x_vals.max()), float(y_vals.min()), float(y_vals.max())

        for s in new_series.keys():
            new_series[s] = new_series[s].astype('float')
            if resize_x:
                new_series[s][:, 0] = (new_series[s][:, 0]-xmin)/(xmax-xmin)
            if resize_y:
                new_series[s][:, 1] = (new_series[s][:, 1]-ymin)/(ymax-ymin)

        # Slope correction for simpler series
        if len(new_series) == 1 and len(new_series[0]) <= 50:
            for s in new_series.keys():
                x = new_series[s]
                for i in range(len(x)-1):
                    if abs(x[i+1, 1]-x[i, 1])/abs(x[i+1, 0]-x[i, 0]+1e-06) < 0.05:
                        upper, lower = np.argmax([x[i, 1], x[i+1, 1]]), np.argmin([x[i, 1], x[i+1, 1]])
                        x[i+upper, 1] = x[i+lower, 1]

                    elif abs(x[i+1, 1]-x[i, 1])/abs(x[i+1, 0]-x[i, 0]+1e-06) > 0.95:
                        righter, lefter = np.argmax([x[i, 0], x[i+1, 0]]), np.argmin([x[i, 0], x[i+1, 0]])
                        x[i+righter, 0] = x[i+lefter, 0]

        return new_series, warnings
    
    def run(self, output_dir='outputs/'):
        cropped, withaxes, xcorr, ycorr = self.ext.run_for_text()
        series = self.extract_series(cropped, 2)
        boxes = self.get_text_data(cropped, withaxes)
        new_series, warns = self.find_points(withaxes, boxes, series, xcorr, ycorr)

        # Save files
        for s in new_series.keys():
            data = new_series[s]
            data_dct = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1]})
            data_dct.to_csv(os.path.join(output_dir, 'data/', f'{self.img_name}_{s}.csv'), index=False)

        return new_series, warns
    
    def plot_outputs(self, new_series, output_dir='outputs/'):
        plt.figure(figsize=(10, 8))
        for k, s in new_series.items():
            s = np.asarray(s).reshape(-1, 2)
            plt.plot(s[:, 0], s[:, 1], linewidth=3, label=str(k))
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'plots/' f'{self.img_name}.png'))