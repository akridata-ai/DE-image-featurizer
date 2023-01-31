import sys
import pathlib
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow_hub as hub
from tensorflow.keras import layers
from ..featurize.descriptor import HOG, LBP, ORB, SIFT
class ImageFeatureEvaluator:
    """
    Provides the functionality to compare different image featurizers using KNN accuracy scores.
    1. The dataset is divided into train and test sets.
    2. Features are computed using the featurizer provided.
    3. Comparative analysis is shown on different numbers of reduced dimensions using PCA.

    Parameters
    ----------

    data_dir : string
        Provides the path to parent directory where data is present
    featurizer_selected : string
        Hyperlink to the Image featurizer module
    batch_size: integer
        Size of batches of data.
    structure: boolean
        Whether the directory is further divided into train and test folders.


    Examples
    ----------
    Select the path to the dataset where the training and testing set are divided as given below
    if structure==False

        ```
        main_directory/
        ......class_a/
        .........a_image1
        .........a_img2
        ......class_b/
        .........b_img1
        .........b_img2
        ```
    else

        ```
        main_directory/
        ....train/
        ......class_a/
        .........a_image1
        .........a_img2
        ......class_b/
        .........b_img1
        .........b_img2
        ....test/
        ......class_a/
        .........a_image1
        .........a_img2
        ......class_b/
        .........b_img1
        .........b_img2
        ```

    Let's try on a simple tensorflow dataset.
    >>> dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    >>> archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
    ...
    >>> data_dir = pathlib.Path(archive).with_suffix('')
    >>> fan = ImageFeatureEvaluator(data_dir,'https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/5',32,False)

    >>> train_ds, _ = fan.load_data()
    Found ...
    >>> featurizer, preprocessor = fan.load_featurizer()
    >>> train_features, train_labels = fan.extract_features(featurizer,preprocessor,train_ds,True)
    >>> train_features.shape
    (2936, 256)
    >>> fan = ImageFeatureEvaluator(data_dir,'hog',32,False)
    >>> featurizer, preprocessor = fan.load_featurizer()
    >>> train_features, train_labels = fan.extract_features(featurizer,preprocessor,train_ds,True)
    >>> train_features.shape
    (2936, 1568)
    """

    def __init__(self, data_dir, featurizer_selected, batch_size, structure):
        self.data_path = data_dir
        self.featurizer_selected = featurizer_selected
        self.batch_size = batch_size
        self.structure = structure

    def load_data(self):
        """
        Creates training and testing dataset from a directory of images where labels are inferred from directory structure

        Returns
        -------
        train_ds :  A `tf.data.Dataset` object
            The training dataset

        test_ds :  A `tf.data.Dataset` object
            The testing dataset
        """
        if self.structure:
            train_ds = tf.keras.utils.image_dataset_from_directory(self.data_path + '/train',
                                                                   seed=123, batch_size=self.batch_size)
            val_ds = tf.keras.utils.image_dataset_from_directory(self.data_path + '/test',
                                                                 seed=123, batch_size=self.batch_size)
        else:
            train_ds = tf.keras.utils.image_dataset_from_directory(self.data_path, validation_split=0.2,
                                                                   subset='training',
                                                                   seed=123, batch_size=self.batch_size)
            val_ds = tf.keras.utils.image_dataset_from_directory(self.data_path, validation_split=0.2,
                                                                 subset='validation',
                                                                 seed=123, batch_size=self.batch_size)
        return train_ds, val_ds

    def evaluate(self):
        """
          Consolidates all the below defined steps of the process of image featurizer evaluation:
          1. Loads the train and test set
          2. Loads the featurizer object
          3. Calculates the features for the train and test set
          4. Computes the results dataframe containing the accuracy scores.

        Returns
        -------
        results_frame : A pandas.DataFrame object
            Pandas dataframe which consists of the evaluation statistics.

        """

        train_ds, val_ds = self.load_data()
        featurizer, preprocessor = self.load_featurizer()
        train_features, train_labels = self.extract_features(featurizer, preprocessor, train_ds, True)
        test_features, test_labels = self.extract_features(featurizer, preprocessor, val_ds, False)
        results_frame = self.accuracy_stats(train_features, test_features, train_labels, test_labels)
        return results_frame

    def feature_loader_and_provide_score(self, train_features_path, train_labels_path,
                                         test_features_path, test_labels_path):
        """
        Computes the stats related to evaluation metric by using pre computed features


        Parameters
        ----------

        train_features_path : string
            Provides the path to the precomputed features for training data.

        train_labels_path : string
            Provides the path to the labels corresponding to training data.

        test_features_path : string
            Provides the path to the precomputed features for the testing data.

        test_labels_path : string
            Provides the path to the labels corresponding to testing data.

        Returns
        -------

        results_frame : A pandas.DataFrame object
            Pandas dataframe which consists of the evaluation statistics.

        """

        train_features = np.load(train_features_path)
        train_labels = np.load(train_labels_path)
        test_features = np.load(test_features_path)
        test_labels = np.load(test_labels_path)
        results_frame = self.accuracy_stats(train_features, test_features, train_labels, test_labels)

        return results_frame

    def load_featurizer(self):
        """
        Instantiates the model from TF-hub library or from the featurizers present in the DE-image-featurizer repository
        and also creates a preprocessor to standardise the input.

        Returns
        -------

        image_featurizer : function
            The image featurizer model to be used for extracting features.

        preprocessor : function
            The basic preprocessor function required to standardise input
        """

        if self.featurizer_selected.lower() == 'hog':
            hog = HOG()
            image_featurizer = hog.fit_transform
        elif self.featurizer_selected.lower() == 'lbp':
            lbp = LBP()
            image_featurizer = lbp.fit_transform
        elif self.featurizer_selected.lower() == 'orb':
            orb = ORB()
            image_featurizer = orb.fit_transform
        elif self.featurizer_selected.lower() == 'sift':
            sift = SIFT()
            image_featurizer = sift.fit_transform

        else:
            image_featurizer = hub.KerasLayer(self.featurizer_selected,
                                              input_shape=(224, 224, 3), trainable=False)
        preprocessor = tf.keras.Sequential([layers.Resizing(224, 224),
                                            layers.Rescaling(1. / 255)])

        return image_featurizer, preprocessor

    def extract_features(self, image_featurizer, preprocessor, input_data, train):
        """
        Computes the features for the input data using the image featurizer

        Parameters
        ----------

        image_featurizer : function
            The image featurizer model imported from TF-hub

        preprocessor : function
        The preprocessing  function comprising of rescaling and resizing.

        input_data : A `tf.data.Dataset` object.
            The data containing images as elements

        saving_path : string
            The path to save extracted features

        train : boolean
            To distinguish between training and testing set

        Returns
        -------

        features_array : numpy.ndarray of shape (train_data_size,num_of_features)
            Numpy array containing extracted features

        input_labels : numpy.ndarray of shape (train_data_size,)
            Numpy array containing corresponding labels

        """

        features_array = None
        input_labels = None

        for image_batch, labels_batch in tqdm(input_data):
            preprocessed_image_batch = preprocessor(image_batch)
            image_batch_features = image_featurizer(preprocessed_image_batch)
            image_batch_features = image_batch_features if isinstance(image_batch_features,
                                                                      list) else image_batch_features
            if features_array is None:
                features_array = image_batch_features
            else:
                features_array = np.vstack((features_array, image_batch_features))
            if input_labels is None:
                input_labels = labels_batch
            else:
                input_labels = np.concatenate([input_labels, labels_batch.numpy()])
        if train:
            np.save('train_features.npy', features_array)
            np.save('train_labels.npy', input_labels)
        else:
            np.save('test_features.npy', features_array)
            np.save('test_labels.npy', input_labels)

        features_array = features_array.squeeze()

        if features_array.ndim >= 3:
            features_array = features_array.reshape(*features_array.shape[:(-features_array.ndim + 1)], -1)
        return features_array, input_labels

    def accuracy_scores(self, train_features, train_labels,
                        test_features, test_labels,
                        n_neighbors):
        """
        Applies KNeighborsClassifier on the input data and return the accuracy obtained on training and testing sets.

        Parameters
        ----------

        train_features : numpy.ndarray of shape (train_data_size,num_of_features)
                The extracted features for training data

        train_labels : numpy.ndarray of shape (train_data_size,)
                The labels for training data

        test_features : numpy.ndarray of shape (test_data_size,num_of_features)
                The extracted features for testing data

        test_labels : numpy.ndarray of shape (test_data_size,)
                The labels for testing data

        n_neighbors : numpy.ndarray
            Number of neighbors to consider for classification

        Returns
        -------

        train_accuracy : float
            The accuracy computed on training set

        test_accuracy : float
            The accuracy computed on testing set

        """

        classifier = KNeighborsClassifier(n_neighbors, weights="distance")
        # Fit the classifier on the Train Data
        classifier.fit(train_features, train_labels)
        # Train Predictions
        y_pred_train = classifier.predict(train_features)
        # Test Predictions
        y_pred_test = classifier.predict(test_features)
        # Train Accuracy Score
        train_accuracy = accuracy_score(train_labels, y_pred_train)
        # Test Accuracy Score
        test_accuracy = accuracy_score(test_labels, y_pred_test)

        return train_accuracy, test_accuracy

    def accuracy_stats(self, train_features, test_features,
                       train_labels, test_labels):
        """
        Applies PCA on the data over several number of dimensions and return a consolidated evaluation statistics.

        Parameters
        ----------

        train_features : numpy.ndarray of shape (train_data_size,num_of_features)
            The extracted features for training data

        train_labels : numpy.ndarray of shape (train_data_size,)
            The labels for training data

        test_features : numpy.ndarray of shape (test_data_size,num_of_features)
            The extracted features for testing data

        test_labels : numpy.ndarray of shape (test_data_size,)
            The labels for testing data

        Returns
        -------

        results_frame: A pandas.DataFrame object
            Pandas dataframe which consists of the evaluation statistics.


        """

        # A grid-search for PCA.
        train_accuracies = []
        test_accuracies = []
        dimensions = [2, 8, 16, 32, 64, 128]
        for n in dimensions:
            pca = PCA(n_components=n)
            reduced_train_features = pca.fit_transform(train_features)
            reduced_test_features = pca.transform(test_features)
            train_accuracy, test_accuracy = self.accuracy_scores(reduced_train_features, train_labels,
                                                                 reduced_test_features, test_labels, 5)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
        results_frame = pd.DataFrame(list(zip(dimensions, train_accuracies, test_accuracies)),
                                     columns=['Dimensions', 'Train_Accuracy', 'Test_Accuracy'])

        return results_frame


if __name__ == "__main__":
    import doctest
    print("Running doctests...")
    doctest.testmod(optionflags=doctest.ELLIPSIS)