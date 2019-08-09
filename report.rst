================================================================
Towards a Universal Classifier for Crystallographic Space Groups
================================================================
Addressing Space Group Imbalance in Large Crystallographic Datasets for Training a Neural Network.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:Contributors: Emily Costa\ :sup:`ab`, Alvin Tan\ :sup:`cd`, Yuya Kawakami\ :sup:`ae`, Shuto Araki\ :sup:`af`

In machine learning, imbalanced datasets can cause bias in the results of the algorithm. In order to achieve excellent performance, the datasets are processed before being run through the machine learning algorithm. Typically, in solving a neural network, more data is better. However, if that dataset is large enough and unnecessary, it is possible to reduce the dataset without greatly compromising the performance of the trained system. The two issues our group addressed in our research are (**1**) developing a machine learning algorithm for space group classification of CBED data and (**2**) implementing proper machine learning techniques to overcome data imbalance and show how it affects the performance of the machine learning algorithm. 

**Machine Learning for Space Group Classification of CBED data**

Shuto talks about resNET or defectNET or whatever

How did we measure performance?

**Addressing the Data Imbalance**

Though the machine learning algorithm was significant, our main focus of this project was the balance the dataset to optimize the performance and accurancy of the neural network.

.. image:: https://raw.githubusercontent.com/emilyjcosta5/datachallenge2/master/distributions/graphs/distributions_bar_log.png
  :width: 400
  :alt: Original Distribution

One of the issues we faced was a difference in the composition of the training dataset and the testing dataset. This means that some space groups that were well represented in the training set did not appear in the testing set, while the testing set also contained space groups that were not in the training set (Figure [][]). The former is not very disruptive, but the latter proves rather problematic, as it's hard to characterize something if it's never been seen before. Thus, we wanted to redistribute all of our data between our training, development, and testing datasets such that the representation of each space group is proportional across all three datasets, giving similar (albeit still nonuniform) distributions of space group samples across all three datasets.

The code to redistribute the data across our three datasets can be found in processing/make_dists_similar_summit.py. The gist is that after creating the three HDF5 files to hold our new datasets, we iterate through all of the data we have available and pseudorandomly distribute them between our three datasets. This theoretically results in similar representations of the space groups across all three datasets. We can also encourage one dataset to be larger than the other by adjusting the structure of the random selection. In our case, we wanted the training set to be about seven times as large as the development and testing sets, which was achieved by simply making it seven times as likely to send data to the training set than to the development set or to the testing set. This 7-1-1 ratio was selected by inspecting the current training, development, and testing datasets and using a ratio similar to the relative sizes of those. Of course, this method may result in poor distribution of sparse space groups, so for all space groups that had less than 30 samples total, we copied every sample into each of our datasets. Thus, the scantest pseudorandom redistribution would be that of a space group with 30 samples, which would give roughly 23 samples in the training set, 3 samples in the development set, and 3 samples in the testing set. Our resulting datasets' space group distributions can be seen in Figure [][].

Emily adds distribution post Alvin

Yuya talks about SMOTE

Yuya adds example image of GANs generated psuedo-data

Emily adds image of balanced dataset

References
~~~~~~~~~~
https://smc-datachallenge.ornl.gov/challenges-2019/challenge-2-2019/

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
