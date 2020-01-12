#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:58:43 2019

@author: shane
"""

from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn import model_selection
from decimal import Decimal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

#task1
def splitting_and_counting(product_images):
    print("\n****************Task 1********************")
    
    sneakers = product_images[product_images.label == 0]
    
    plt.imshow(sneakers.values[0][1:].reshape(28, 28))
    plt.show()
    
    ankle_boots = product_images[product_images.label == 1]
    
    plt.imshow(ankle_boots.values[0][1:].reshape(28, 28))
    plt.show()
    
    print("Num of Sneakers:", len(sneakers))
    print("Num of Ankle Boots:", len(ankle_boots))
        
#task2
def perceptron(product_images, num_samples, num_splits):
    print("\n****************Task 2********************")
    training_time = []
    prediction_time = []
    prediction_accuracy = []
        
    kf = model_selection.KFold(n_splits=num_splits, shuffle=True)
    
    product_images = product_images.head(num_samples)
        
    for train_index,test_index in kf.split(product_images.values):
        clf1 = linear_model.Perceptron() 
    
        start = time.time()
        clf1.fit(product_images.values[train_index], product_images.label[train_index])
        end = time.time()
        training_time.append(end - start)
        
        start = time.time()
        prediction1 = clf1.predict(product_images.values[test_index])
        print("Perceptron Prediction Labels:\n", prediction1)
        end = time.time()
        prediction_time.append(end - start)
        
        score1 = metrics.accuracy_score(product_images.label[test_index], prediction1)
        
        confusion = metrics.confusion_matrix(product_images.label[test_index], prediction1)
        
        prediction_accuracy.append(score1)
        print("Perceptron accuracy score: ", score1)
        print("Confusion Matrix: \n", confusion)
        
        print()
        
    print("----------Training Time-----------")
    print("Max time value: ", np.max(training_time))
    print("Min time value: ", np.min(training_time))
    print("Avg time value: ", np.mean(training_time))
    print()
    
    print("----------Prediction Time-----------")
    print("Max time value: ", np.max(prediction_time))
    print("Min time value: ", np.min(prediction_time))
    print("Avg time value: ", np.mean(prediction_time))
    print()
    
    print("-------Perceptron Prediction Accuracy-------")
    print("Max accuracy score: ", np.max(prediction_accuracy))
    print("Min accuracy score: ", np.min(prediction_accuracy))
    print("Avg accuracy score: ", np.mean(prediction_accuracy))
    print()

#task3
def support_vector_machine(product_images, num_samples, num_splits):
    print("\n****************Task 3********************")
    linear_training_time = []
    linear_prediction_time = []
    rbf_training_time = []
    rbf_prediction_time = []
    linear_prediction_accuracy = {'1e-1': [], '1e-2': [], '1e-3': [], '1e-4': [], '1e-5': []}
    rbf_prediction_accuracy = {'1e-1': [], '1e-2': [], '1e-3': [], '1e-4': [], '1e-5': []}
    
    gamma_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        
    kf = model_selection.KFold(n_splits=num_splits, shuffle=True)
    
    product_images = product_images.head(num_samples)
       
    for train_index,test_index in kf.split(product_images.values):
        for i in gamma_list:
            print("--------------","{:.0e}".format(Decimal(i)), "----------------")
            clf1 = svm.SVC(kernel="linear", gamma=i)       
            clf2 = svm.SVC(kernel="rbf", gamma=i)
        
            start = time.time()
            clf1.fit(product_images.values[train_index], product_images.label[train_index])
            end = time.time()
            linear_training_time.append(end - start)
            
            start = time.time()
            prediction1 = clf1.predict(product_images.values[test_index])
            print("Linear Prediction Labels:\n", prediction1)
            end = time.time()
            linear_prediction_time.append(end - start)
        
            start = time.time()
            clf2.fit(product_images.values[train_index], product_images.label[train_index])
            end = time.time()
            rbf_training_time.append(end - start)
            
            start = time.time()
            prediction2 = clf2.predict(product_images.values[test_index])
            print("RBF Prediction Labels:\n", prediction2)
            end = time.time()
            rbf_prediction_time.append(end - start)
            
            score1 = metrics.accuracy_score(product_images.label[test_index], prediction1)
            score2 = metrics.accuracy_score(product_images.label[test_index], prediction2)
            
            confusion1 = metrics.confusion_matrix(product_images.label[test_index], prediction1)
            confusion2 = metrics.confusion_matrix(product_images.label[test_index], prediction2)
            
            linear_prediction_accuracy['{:.0e}'.format(Decimal(i))].append(score1)
            rbf_prediction_accuracy['{:.0e}'.format(Decimal(i))].append(score2)
            print("SVM with Linear kernel accuracy score: ", score1)
            print("SVM with RBF kernel accuracy score: ", score2)
            print("Linear Kernel Confusion Matrix: \n", confusion1)
            print("RBF Kernel Confusion Matrix: \n", confusion2)
            
            print()
    
    print("-------Linear Training Time--------")
    print("Max time value: ", np.max(linear_training_time))
    print("Min time value: ", np.min(linear_training_time))
    print("Avg time value: ", np.mean(linear_training_time))
    print()
    
    print("-------Linear Prediction Time--------")
    print("Max time value: ", np.max(linear_prediction_time))
    print("Min time value: ", np.min(linear_prediction_time))
    print("Avg time value: ", np.mean(linear_prediction_time))
    print()
    
    print("---------RBF Training Time---------")
    print("Max time value: ", np.max(rbf_training_time))
    print("Min time value: ", np.min(rbf_training_time))
    print("Avg time value: ", np.mean(rbf_training_time))
    print()
    
    print("---------RBF Prediction Time---------")
    print("Max time value: ", np.max(rbf_prediction_time))
    print("Min time value: ", np.min(rbf_prediction_time))
    print("Avg time value: ", np.mean(rbf_prediction_time))
    print()
    
    for i in gamma_list:
        print("-------Linear {:.0e}".format(Decimal(i)), "Prediction Accuracy--------")
        print("Max accuracy score: ", np.max(linear_prediction_accuracy['{:.0e}'.format(Decimal(i))]))
        print("Min accuracy score: ", np.min(linear_prediction_accuracy['{:.0e}'.format(Decimal(i))]))
        print("Avg accuracy score: ", np.mean(linear_prediction_accuracy['{:.0e}'.format(Decimal(i))]))
        print()
    for i in gamma_list:
        print("--------RBF {:.0e}".format(Decimal(i)), "Prediction Accuracy---------")
        print("Max accuracy score: ", np.max(rbf_prediction_accuracy['{:.0e}'.format(Decimal(i))]))
        print("Min accuracy score: ", np.min(rbf_prediction_accuracy['{:.0e}'.format(Decimal(i))]))
        print("Avg accuracy score: ", np.mean(rbf_prediction_accuracy['{:.0e}'.format(Decimal(i))]))
        print()
    
    
def main():
    product_images = pd.read_csv("product_images.csv")
    num_samples = int(input("Enter the num of samples (0-14000): "))
    num_splits = int(input("Enter the num of splits: "))
    splitting_and_counting(product_images)
    perceptron(product_images, num_samples, num_splits)
    support_vector_machine(product_images, num_samples, num_splits)

main()