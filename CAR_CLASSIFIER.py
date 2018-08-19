from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import pickle
import cv2
import glob
import time
from helpers import *

class CAR_CLASSIFIER:
    def __init__(self, data):
        self.color_space = 'RGB'
        self.spatial_size = (16, 16)
        self.hist_bins = 16
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = 0
        self.car_data = data['car_data']
        self.noncar_data = data['noncar_data']
        self.car_sp_hist = None
        self.non_car_sp_hist = None
        self.car_features = None
        self.non_car_features = None
        self.X = None
        self.y = None
        self.X_scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.spatial_feat = True
        self.hist_feat = True
        self.hog_feat = True
    
    def data_split(self, split_size = 0.2, bDebuggingFlag = False):
        rand_state = np.random.randint(0, 100)
        if bDebuggingFlag == True:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.scaled_X, self.y, test_size = split_size, random_state=rand_state)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size = split_size, random_state=rand_state)
        
#         car_split    = len(self.car_features) * 0.2
#         notcar_split = len(self.non_car_features) * 0.2

#         self.X_test = np.vstack((self.scaled_X[:int(car_split)],
#             self.scaled_X[len(self.car_features):(len(self.car_features) + int(notcar_split))]))

#         self.y_test = np.hstack((self.Y[:int(car_split)],
#             self.Y[len(self.car_features):(len(self.car_features) + int(notcar_split))]))

#         self.X_train = np.vstack((self.scaled_X[int(car_split):len(self.car_features)],
#             self.scaled_X[(len(self.car_features) + int(notcar_split)):]))

#         self.y_train = np.hstack((self.Y[int(car_split):len(self.car_features)],
#             self.Y[(len(self.car_features) + int(notcar_split)):]))

    def set_feature_and_labels(self):
        print("Starting HOG feature extraction...")
        t = time.time()
        
        self.car_features = extract_features(self.car_data,
                color_space=self.color_space,
                orient=self.orient,
                pix_per_cell=self.pix_per_cell,
                cell_per_block=self.cell_per_block,
                hog_channel=self.hog_channel)

        self.non_car_features = extract_features(self.noncar_data,
                color_space=self.color_space,
                orient=self.orient,
                pix_per_cell=self.pix_per_cell,
                cell_per_block=self.cell_per_block,
                hog_channel=self.hog_channel)

        t2 = time.time()
        print(round(t2-t, 2), "Seconds to extract HOG features...")
        
        self.X = np.vstack((self.car_features, self.non_car_features)).astype(np.float64)
        self.y = np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.non_car_features))))
        
        print("Using:", orient,"orientations", pix_per_cell, 
              "pixels per cell and", cell_per_block,"cells per block")
        print("Feature vectors shape:",self.X.shape)
    
    def set_feature_and_labels_with_color_settings(self, oSettings):
        print("Starting HOG feature extraction...")
        t = time.time()
        
        self.color_space = oSettings['color_spacce']
        self.spatial_size = oSettings['spatial_size']
        self.hist_bins = oSettings['num_hist_bins']
        self.orient = oSettings['orient']
        self.pix_per_cell = oSettings['pixel_per_cell']
        self.cell_per_block = oSettings['cell_per_block']
        self.hog_channel = oSettings['hog_channel']
        
        self.car_features = extract_features(self.car_data,
                                            color_space=self.color_space,
                                            spatial_size = self.spatial_size,
                                            hist_bins = self.hist_bins,
                                            orient=self.orient,
                                            pix_per_cell=self.pix_per_cell,
                                            cell_per_block=self.cell_per_block,
                                            hog_channel=self.hog_channel,
                                            spatial_feat = self.bSpatial_feat,
                                            hist_feat = self.bHist_feat,
                                            hog_feat = self.bHog_feat)

        self.non_car_features = extract_features(self.noncar_data,
                                                color_space=self.color_space,
                                                spatial_size = self.spatial_size,
                                                hist_bins = self.hist_bins,
                                                orient=self.orient,
                                                pix_per_cell=self.pix_per_cell,
                                                cell_per_block=self.cell_per_block,
                                                hog_channel=self.hog_channel,
                                                spatial_feat = self.bSpatial_feat,
                                                hist_feat = self.bHist_feat,
                                                hog_feat = self.bHog_feat)

        t2 = time.time()
        print(round(t2-t, 2), "Seconds to extract HOG features...")
        
        self.X = np.vstack((self.car_features, self.non_car_features)).astype(np.float64)
        self.y = np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.non_car_features))))
        
        print("Using:", orient,"orientations", pix_per_cell, 
              "pixels per cell and", cell_per_block,"cells per block")
        print("Feature vectors shape:",self.X.shape)
        
    def scale_features(self):
        self.X_scaler = StandardScaler().fit(self.X)
        self.scaled_X = self.X_scaler.transform(self.X)
        
    def train_SVC(self):
        svc = LinearSVC()
        print("Training SVC...")
        t=time.time()

        svc.fit(self.X_train, self.y_train)

        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')

        self.svc = svc
    
    def get_SVC_accuracy(self):
        print("Test Accuracy of SVC = ", round(self.svc.score(self.X_test, self.y_test), 4))
    
    def get_prediction(self, testSets, testLabels):
        total_num = len(testLabels)
        correct_count = 0
        
        predicted_results = []
        for i in range(total_num):
            res = self.svc.predict(testSets[i])
            predicted_results.append(res)
            if res == testLabels[i]:
                correct_count += 1
        
        accuracy = float(correct_count/total_num) * 100
        print("SVC prediction: ", predicted_results)
        print("Actual        : ", testLabels)
        print("Prediction accuracy:  ", accuracy, "%")
    
    def save_pickle_files(self):
        print('Saving data to pickle file...')
        with open('processed_hog_data.p', 'wb') as pfile:
                pickle.dump({   
                    'svc':self.svc,
                    'X': self.X,
                    'y': self.y,
                    'X_scaler': self.X_scaler,
                    'X_train': self.X_train,
                    'y_train': self.y_train,
                    'X_test': self.X_test,
                    'y_test': self.y_test,
                    'car_sp_hist': self.car_sp_hist,
                    'noncar_sp_hist': self.noncar_sp_hist
                },pfile, pickle.HIGHEST_PROTOCOL)

        print('Data cached in pickle file.')

    def load_model(self):
        with open('processed_hog_data.p', mode = 'rb') as file:
            processed_data = pickle.load(file)
        self.svc = processed_data['svc']
        self.X = processed_data['X']
        self.y = processed_data['y']
        self.X_scaler = processed_data['X_scaler']
        self.X_train = processed_data['X_train']
        self.y_train = processed_data['y_train']
        self.X_test = processed_data['X_test']
        self.y_test = processed_data['y_test']
#         self.car_sp_hist = processed_data['car_sp_hist']
#         self.noncar_sp_hist = processed_data['noncar_sp_hist']