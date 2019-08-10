=============================================================================
Towards a Universal Classifier for Crystallographic Space Groups\ :sup:`1`
=============================================================================

Addressing Space Group Imbalance in Large Crystallographic Datasets for Training a Neural Network.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:Contributors: Emily Costa\ :sup:`ab`, Alvin Tan\ :sup:`cd`, Yuya Kawakami\ :sup:`ae`, Shuto Araki\ :sup:`af`
:Video: http://bit.ly/smc_conference

In machine learning, imbalanced datasets can cause bias in the results of the algorithm. In order to achieve excellent performance, the datasets are processed before being run through the machine learning algorithm. Typically, in solving a neural network, more data is better. However, if that dataset is large enough and unnecessary, it is possible to reduce the dataset without greatly compromising the performance of the trained system. The two issues our group addressed in our research are (**1**) developing a machine learning algorithm for space group classification of CBED data and (**2**) implementing proper machine learning techniques to overcome data imbalance and show how it affects the performance of the machine learning algorithm. 

Machine Learning for Space Group Classification of CBED data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Model evaluation benchmark with ResNet-50**

Deep Residual Network with 50 layers, commonly referred to as the ResNet-50\ :sup:`2`, to benchmark our classification performance. The ResNet-50 is one of the most popular convolutional neural networks for image classification tasks. 

We compared results before and after the data imbalance techniques mentioned below are applied, with everything else constant. Due to our time constraints, we could only implement this one model. Hence, this project is mainly focused on the data imbalance techniques rather than the development of specific machine learning models. In this project, the ResNet-50 rather serves as a benchmark for further exploration of different models in the future.

Model specification: 

:Batch Size: 128
:Epochs: 90
:Learning Rate: 0.01
:Momentum: 0.9
:Weight Decay: 0.00005
:Loss Function: Cross Entropy


(**1**) **Before Redistribution**

The evaluation accuracy suffered from the heavy data imbalance mentioned below and ended up being only about 2.7% accuracy. While this result is better than random chance (1/230 ~ 0.43%), it barely learned any patterns in the data partially because some images in the test dataset included classes that do not exist in the training dataset.

(**2**) **After Redistribution**

The evalution accuracy improved to 23.5%, which is close to 10x higher than the non-processed data. While this accuracy is still not high enough to be a useful classifier, it shows the effectiveness of the data imbalance techniques explained in the next section.

Furthermore, the model is by no means properly tuned (and therefore has a signicant room for improvement), but the redistribution of the imbalanced classes and SMOTE mentioned below shows significant improvement. The code for this experiment is available under the :code:`pytorch` directory of this repository.

Addressing the Data Imbalance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main focus of this project was to balance the dataset to optimize the performance and accurancy of the neural network. The overall representation of datasets exhibited a significant data imbalance. The average space group representation of the overall dataset was 814 images per space group, while the 25th percentile average was only 23 images per space group. The distribution of the original data can be seen in Figure 1. The balancing of this data occurred in two stages, (**1**) redistributing the original dataset in order to represent all space groups in the training dataset for the neural network and (**2**) balancing the overall data using various techniques to represent space groups evenly in the training dataset in order to minimize bias in the neural network. 

.. image:: https://raw.githubusercontent.com/emilyjcosta5/datachallenge2/master/distributions/graphs/distributions_bar_log.png
  :width: 2000
  :alt: Figure 1. Original Distribution

Figure 1. Original Distribution

**Redistributing the Original Dataset**

One of the issues we faced was a difference in the composition of the training dataset and the testing dataset. This means that some space groups that were well represented in the training set did not appear in the testing set, while the testing set also contained space groups that were not in the training set (Figure 1). The former is not very disruptive, but the latter proves rather problematic, as it's hard to characterize something if it's never been seen before. Thus, we wanted to redistribute all of our data between our training, development, and testing datasets such that the representation of each space group is proportional across all three datasets, giving similar (albeit still nonuniform) distributions of space group samples across all three datasets.

The code to redistribute the data across our three datasets can be found in processing/make_dists_similar_summit.py. The gist is that after creating the three HDF5 files to hold our new datasets, we iterate through all of the data we have available and pseudorandomly distribute them between our three datasets. This theoretically results in similar representations of the space groups across all three datasets. We can also encourage one dataset to be larger than the other by adjusting the structure of the random selection. In our case, we wanted the training set to be about seven times as large as the development and testing sets, which was achieved by simply making it seven times as likely to send data to the training set than to the development set or to the testing set. This 7-1-1 ratio was selected by inspecting the current training, development, and testing datasets and using a ratio similar to the relative sizes of those. Of course, this method may result in poor distribution of sparse space groups, so for all space groups that had less than 30 samples total, we copied every sample into each of our datasets until each space group had at least 30 samples. Thus, the scantest pseudorandom redistribution would be that of a space group with 30 samples. Our resulting datasets' space group distributions can be seen in Figure 2. Now, our overall average amount of samples per space group is 839, while the 25th percentile average increased to 90 images per space group.

.. image:: https://raw.githubusercontent.com/emilyjcosta5/datachallenge2/master/distributions/functions/redistributions_bar_log.png
  :width: 2000
  :alt: Figure 2. Redistributed Distribution

Figure 2. Redistributed Distribution

**Overall Balancing**

To further address the data imbalance, a combination of two techniques was used.

(**1**) **Under-sampling**, which deletes instances from any classes that might be in an over-represented space groups. Several space groups exceeded

(**2**) **Over-sampling**, synthetic data was generated to compensate for under-represented space groups. 

As mentioned, an imbalanced dataset can be detrimental to the performance of a machine learning algorithm. Over-sampling of minority classes with the creation of synthetic minority class data is one method to deal with an imbalanced dataset. To this end, we propose using SMOTE (Synthetic Minority Over-Sampling Technique) \ :sup:`2`. With SMOTE, synthetic samples are generated using by taking the k nearest neighobors of a sample, and generating a random point along the line segment  between the sample in question and and the nearest neigbhors. Details of SMOTE is outlined in the referenced paper. We used the SMOTE implementation in Python's :code:`imbalanced-learn` package. 

.. image:: https://raw.githubusercontent.com/emilyjcosta5/datachallenge2/master/train/original.png
   :width: 1200

Figure 3. Original Data

.. image:: https://raw.githubusercontent.com/emilyjcosta5/datachallenge2/master/train/generated.png
   :width: 1500

Figure 4. Synthetic Data

The above images are examples of a SMOTE generated data and the original data from which SMOTE was generated. In the above example, 10 samples of images in Space Group 2 were given to SMOTE to generate 5 synthetic sample. 2 of the original data and 4 of the generated data is shown as an example.  Due to the heavy data imbalance in the dataset and time constraints, it was challenging to increase the model accuracy and took significant amount of engineering effort in order to feed all the data properly. Even after 90 epochs, the model performed very poorly with the evaluation accuracy still stayed at around 2%. With the SMOTE, the evaluation accuracy went up to about 23%, which is a significant improvement but not high enough to be useful.

Future Work 
~~~~~~~~~~~
The SMOTE implementation in the :code:`imbalanced-learn` package allows users to specify the number of synthetic data to generate via a python dictionary. Since this dictates the degree to which we oversample, this is a critical hyperparameter to tune. Furthermore, the number of neighbors that SMOTE uses to generate synthetic data can be specified as an argument (We have used 6 in our example). Further work will include tuning these parameters. 

Future work also includes exploring more models made for Crystallography classification, such as DefectNet created by the Pycroscopy\ :sup:`4`.

References
~~~~~~~~~~
[1] https://smc-datachallenge.ornl.gov/challenges-2019/challenge-2-2019/

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for imagerecognition.CoRR,abs/1512.03385. Retrieved from http://arxiv.org/abs/1512.03385

[4] Chawla, N. V., K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer. "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research 16 (2002): 321-57. doi:10.1613/jair.953.

[4]  Pycroscopy: Scientific analysis of nanoscale materials imaging data, https://pycroscopy.github.io/pycroscopy/about.html

Affiliations
~~~~~~~~~~~~
\ :sup:`a` Advanced Data and Workflows Group, National Center for Computational Sciences, Oak Ridge, TN 37831, USA

\ :sup:`b` Department of Mathematics, Florida International University, Miami, FL 33199, USA

\ :sup:`c` Department of Electrical and Computer Engineering, Northwestern University, Evanston, IL 60208, USA

\ :sup:`d` Center for Nanophase Materials Sciences, Oak Ridge National Laboratory, Oak Ridge, TN 37831, USA

\ :sup:`e` Department of Mathematics, Computer Science, Grinnell College, Grinnell, IA 50112, USA

\ :sup:`f` Department of Computer Science, DePauw University, Greencastle, IN 46135, USA

Acknowledgements
~~~~~~~~~~~~~~~~
This project was supported in part by an appointment to the Science Education and Workforce Development Programs at Oak Ridge National Laboratory, administered by ORISE through the U.S. Department of Energy Oak Ridge Institute for Science and Education.

This project used resources of the Oak Ridge Leadership Computing Facility (OLCF), which is a DOE Office of Science User Facility and the Compute and Data Environment for Science (CADES) at the Oak Ridge National Laboratory supported by the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.
