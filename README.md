# CenterSurroundOnNeuralNet
Implementation of Center-On and Center-Off Receptive Field filters on a Simple Neural Network

Uses a simple neural network based on the lessons and code from Andrew Ng's coursera course. Uses fmincg to implement backpropagation.
Datasets used were MNIST handwritten digit images, ovarian cancer data from MATLAB's statistics and machine learning toolbox, and arcene dataset from the UCI machine learning respository.

Running main.m will first run the algorithm on each dataset with each filter option (no filter, center-on, and center-off). This may take some time as the algorithm will then run 100 trials of each combination (900 total trials). The algorithm will learn on the first 90% of the dataset, then test its accuracy on the remaining 10%. The accuracy of each trial is stored in the vector called results. This will be a 9x100 matrix with the first three rows being each 100 trials on the MNIST dataset with no filter, center-on, and then center-off filters, then the next two sets of three rows being the ovarian cancer and arcene datasets respectively.
