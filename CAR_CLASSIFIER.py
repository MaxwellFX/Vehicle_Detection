from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import numpy as np
import pickle
import cv2
import time
from helpers import *

class CAR_CLASSIFIER():
    def __init__(self, dataset):
        self.color_space = 'YUV'
        self.spatial_size = (16, 16)
        self.hist_bins = 16
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = 0
        self.spatial_feat = True
        self.hist_feat = True
        self.hog_feat = True
        
        self.car_data = dataset['car_data']
        self.non_car_data = dataset['non_car_data']
        
        self.car_features = None
        self.non_car_features = None
        
        self.cells_per_step = 1
        self.all_heats      = []
        self.window         = 64
        self.heat_thresh    = 3
#         self.scales         = [1, 1.5, 2, 2.5, 4]
#         self.y_start_stops  = [[380, 460], [380, 560], [380, 620], [380, 680], [350, 700]]
        self.scales         = [1, 1.5, 2, 2.5, 3]
        self.y_start_stops  = [[380, 492], [380, 548], [380, 604], [380, 680], [350, 700]]

    def extract_data_features(self, oSettings):
        t=time.time()
        
        self.color_space = oSettings['color_space']
        self.spatial_size = oSettings['spatial_size']
        self.hist_bins = oSettings['hist_bins']
        self.orient = oSettings['orient']
        self.pix_per_cell = oSettings['pix_per_cell']
        self.cell_per_block = oSettings['cell_per_block']
        self.hog_channel = oSettings['hog_channel']

        print("Start feature extraction for cars......")
        self.car_features = extract_features(self.car_data,
                                            color_space = self.color_space,
                                            spatial_size = self.spatial_size,
                                            hist_bins = self.hist_bins, 
                                            orient = self.orient,
                                            pix_per_cell = self.pix_per_cell,
                                            cell_per_block = self.cell_per_block,
                                            hog_channel = self.hog_channel,
                                            spatial_feat = self.spatial_feat,
                                            hist_feat = self.hist_feat,
                                            hog_feat = self.hog_feat)

        print()
        print("Start feature extraction for non-cars......")
        self.non_car_features = extract_features(self.non_car_data,
                                            color_space = self.color_space,
                                            spatial_size = self.spatial_size,
                                            hist_bins = self.hist_bins,
                                            orient = self.orient,
                                            pix_per_cell = self.pix_per_cell,
                                            cell_per_block = self.cell_per_block,
                                            hog_channel = self.hog_channel,
                                            spatial_feat = self.spatial_feat,
                                            hist_feat = self.hist_feat,
                                            hog_feat = self.hog_feat)

        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to extract HOG features...')
        self.X = np.vstack((self.car_features, self.non_car_features)).astype(np.float64)
        
        self.y = np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.non_car_features))))
        print('Feature vectors shape:',self.X.shape)

    def scale_features(self):
        self.X_scaler = StandardScaler().fit(self.X)
        self.scaled_X = self.X_scaler.transform(self.X)

    def data_split(self):
        rand_state = np.random.randint(0, 100)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.scaled_X, self.y, test_size=0.2, random_state=rand_state)

    def train_SVC(self):
        svc = LinearSVC()
        t=time.time()

        svc.fit(self.X_train, self.y_train)

        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')

        self.svc = svc

    def get_accuracy(self):
        print('Test Accuracy of SVC = ', round(self.svc.score(self.X_test, self.y_test), 4))

    def predict(self, img):
        print('SVC predicts: ', self.svc.predict(img))
    
    def predict2(self, X_samples, y_samples):
        pass_count = 0
        for i, img in enumerate(X_samples):
            prediction = self.svc.predict(img)
            if prediction == y_samples[i]:
                pass_count += 1
            print("Prediction: ", prediction, ",  actual: ", y_samples[i])
        
        rate = float(pass_count/len(y_samples))
        print("Prediction correct rate = {:.3f}".format(rate * 100))

    def save_pickle_files(self):
        print('Saving data to pickle file...')
        with open('processed_hog_data.p', 'wb') as pfile:
                pickle.dump({   
                    'svc':self.svc,
                    'car_features': self.car_features,
                    'non_car_features': self.non_car_features,
                    'X_scaler': self.X_scaler,
                    'X_train': self.X_train,
                    'y_train': self.y_train,
                    'X_test': self.X_test,
                    'y_test': self.y_test
                },pfile, pickle.HIGHEST_PROTOCOL)

        print('Data cached in pickle file.')

    def load_data(self):
        with open('processed_hog_data.p', mode = 'rb') as file:
            processed_data = pickle.load(file)
        self.svc = processed_data['svc']
        self.car_features = processed_data['car_features']
        self.non_car_features = processed_data['non_car_features']
        self.X_scaler = processed_data['X_scaler']
        self.X_train = processed_data['X_train']
        self.y_train = processed_data['y_train']
        self.X_test = processed_data['X_test']
        self.y_test = processed_data['y_test']
        print('Data loading complete')
    
    def set_hog_channel(self, option):
        self.hog_channel = option
    
    def set_search_scope(self, aSettings):
        self.scales = []
        self.y_start_stops = []
        for (yStart, yStop, scale) in aSettings:
            self.y_start_stops.append([yStart, yStop])
            self.scales.append(scale)
        print("Done")
    
    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars_rect(self, img, ystart, ystop, scale, cells_per_step = 2, 
                       window = 64, b_shall_all_rect = False):
        rectangles = []

        img = img.astype(np.float32)/255

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, color_space='YCrCb')
        # rescale image if other than 1.0 scale
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        # select colorspace channel for HOG 
        if self.hog_channel == 'ALL':
            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]
        else: 
            ch1 = ctrans_tosearch[:,:, self.hog_channel]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        nfeat_per_block = self.orient * self.cell_per_block**2


        nblocks_per_window = (window // self.pix_per_cell) - self. cell_per_block + 1

        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop : ytop+window, xleft : xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size = self.spatial_size)
                hist_features = color_hist(subimg, nbins = self.hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = self.X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1 or b_shall_all_rect:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    rectangles.append(((xbox_left, ytop_draw+ystart),
                                       (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

        return rectangles
    
    def set_sliding_window_param(self, oSettings):
        self.cells_per_step = oSettings['cells_per_step']
        self.window         = oSettings['window']
        
    def find_cars2(self, img, aSettings, bShowAllRectangles = False, color_option = (255,0,0)):
        rects = []
        draw_img = np.copy(img)
        for (ystart, ystop, scale) in aSettings:
            rect = self.find_cars_rect(img, ystart, ystop, scale, 
                                             cells_per_step = self.cells_per_step, 
                                             window = self.window, 
                                             b_shall_all_rect = bShowAllRectangles)
            rects.append(rect)
        rects2draw = [item for sublist in rects for item in sublist] 
        return draw_boxes(draw_img, rects2draw, color = color_option, thick=2) 
            
    def find_cars(self, img):

        draw_img = np.copy(img)
        img = img.astype(np.float32)/255

        rectangles = []

        for y_start_stop, scale in zip(self.y_start_stops, self.scales):

            img_tosearch = img[y_start_stop[0]:y_start_stop[1],:,:]

            ctrans_tosearch = convert_color(img_tosearch, color_space=self.color_space)

            if scale != 1:
                imshape = ctrans_tosearch.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                             (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // self.pix_per_cell)-1
            nyblocks = (ch1.shape[0] // self.pix_per_cell)-1

            nfeat_per_block = self.orient*self.cell_per_block**2

            # Compute individual channel HOG features for the entire image
            hog1 = get_hog_features(ch1, self.orient,
                    self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, self.orient,
                    self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, self.orient,
                    self.pix_per_cell, self.cell_per_block, feature_vec=False)


            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            nblocks_per_window = (self.window // self.pix_per_cell)-1

            nxsteps = (nxblocks - nblocks_per_window) // self.cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // self.cells_per_step

            i = 0

            for xb in range(nxsteps+1):
                for yb in range(nysteps+1):
                    i += 1

                    if xb == (nxsteps + 1):
                        xpos = ch1.shape[1] - nblocks_per_window
                    else:
                        xpos = xb*self.cells_per_step

                    if yb == (nysteps + 1):
                        ypos = ch1.shape[0] - nblocks_per_window
                    else:
                        ypos = yb*self.cells_per_step


                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

                    if self.hog_channel == 'ALL':
                        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    elif self.hog_channel == '0':
                        hog_features = hog_feat1
                    elif self.hog_channel == '1':
                        hog_features = hog_feat2
                    elif self.hog_channel == '2':
                        hog_features = hog_feat3

                    xleft = xpos*self.pix_per_cell
                    ytop  = ypos*self.pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop+self.window, xleft:xleft+self.window], (64,64))

                    # Get color features
                    spatial_features = bin_spatial(subimg, size=self.spatial_size)
                    hist_features    = color_hist(subimg, nbins=self.hist_bins)

                    img_features = []

                    if self.spatial_feat:
                        img_features.append(spatial_features)
                    if self.hist_feat:
                        img_features.append(hist_features)
                    if self.hog_feat:
                        img_features.append(hog_features)

                    img_features = np.concatenate(img_features).reshape(1, -1)

                    # Scale features and make a prediction
                    test_features   = self.X_scaler.transform(img_features)
                    test_prediction = self.svc.predict(test_features)

                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(self.window*scale)

                    if test_prediction == 1:
                        rectangles.append(((xbox_left, ytop_draw + y_start_stop[0]),
                            (xbox_left + win_draw, ytop_draw + win_draw + y_start_stop[0])))

        return rectangles
    
    def processing_pipeline(self, image):
        draw_image = np.copy(image)

        # Generate heatmap
        heatmap = np.zeros((image.shape[0], image.shape[1]), np.uint8)

        hot_windows = self.find_cars(image)

        heatmap = add_heat(heatmap, hot_windows)

        # Apply threshold to the heatmap
        heatmap = apply_threshold(heatmap, self.heat_thresh)

        # Apply SciPy labeling
        labels = label(heatmap)

        # draw the bounding box on the image 
        draw_image = np.copy(image)
        draw_image = draw_labeled_bboxes(draw_image, labels)

        return draw_image
    
    def set_heat_thresh(self, value):
        self.heat_thresh = value