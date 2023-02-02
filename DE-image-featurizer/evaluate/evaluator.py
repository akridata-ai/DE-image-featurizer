"""
evaluate.py
Provides the functinality to compare different image featurizers
Creates a featurizer from tf-hub or traditional repositoy based models.
Applies PCA to reduce feature dimensions.
Provides KNN accuracy scores for different set of dimensions
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow_hub as hub
from ..featurize.descriptor import HOG, LBP, ORB, SIFT


class ImageFeatureEvaluator:
    """
    Provides the functionality to compare different image featurizers using KNN accuracy scores.
    1. The dataset is divided into train and test sets.
    2. Features are computed using the featurizer provided.
    3. Comparative analysis is shown on different numbers of reduced dimensions using PCA.

    Attributes
    ----------

    batch_size: integer
        Size of batches of data.
    structure: boolean
        Whether the directory is further divided into train and test folders.
    train_ds :  A `tf.data.Dataset` object
            The training dataset
    test_ds :  A `tf.data.Dataset` object
            The testing dataset
    image_featurizer : function
        The image featurizer model to be used for extracting features.
    preprocessor : function
        The basic preprocessor function required to standardise input

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
    >>> import pathlib
    >>> dataset_url = "https://storage.googleapis.com/" \
    "download.tensorflow.org/example_images/flower_photos.tgz"
    >>> archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
    >>> data_dir = pathlib.Path(archive).with_suffix('')
    >>> fan = ImageFeatureEvaluator(data_dir,\
    'https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/5',\
    32,False)
    Found...
    >>> train_features, train_labels = fan._extract_features(True)
    >>> train_features.shape
    (2936, 256)
    >>> fan = ImageFeatureEvaluator(data_dir,'hog',32,False)
    Found...
    >>> train_features, train_labels = fan._extract_features(True)
    >>> train_features.shape
    (2936, 1568)
    """

    def __init__(self, data_dir, featurizer_selected, batch_size=32, structure=False):
        """
        Constructor for the ImageFeatureEvaluator class

        Parameters
        -------------
        data_dir : string
            The path to the image dataset
        featurizer_selected : string
            The image featurizer to be used
        batch_size : int, default=32
            The size of the batch of the data
        structure: boolean , default=False
            Whether the directory is further divided into train and test folders.
        """

        self.batch_size = batch_size
        self.structure = structure
        self._load_data(data_dir)
        self._load_featurizer(featurizer_selected)

    def _load_data(self,data_dir):
        """
        Creates training and testing dataset from a directory
        of images where labels are inferred from directory structure

        Parameters
        -------------
        data_dir : string
            The path to the image dataset
        """

        if self.structure:
            self.train_ds = tf.keras.utils.image_dataset_from_directory(data_dir + '/train',
            seed=123, batch_size=self.batch_size)
            self.val_ds = tf.keras.utils.image_dataset_from_directory(data_dir + '/test',
            seed=123, batch_size=self.batch_size)
        else:
            self.train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
            validation_split=0.2,subset='training',seed=123, batch_size=self.batch_size)
            self.val_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
            validation_split=0.2,subset='validation',seed=123, batch_size=self.batch_size)

    def _load_featurizer(self,featurizer_selected):
        """
        Instantiates the model from TF-hub library or
        from the featurizers present in the DE-image-featurizer repository
        and also creates a preprocessor to standardise the input.

        Parameters
        -------------
        featurizer_selected : string
            The image featurizer to be used
        """

        if featurizer_selected.lower() == 'hog':
            hog = HOG()
            self.image_featurizer = hog.fit_transform
        elif featurizer_selected.lower() == 'lbp':
            lbp = LBP()
            self.image_featurizer = lbp.fit_transform
        elif featurizer_selected.lower() == 'orb':
            orb = ORB()
            self.image_featurizer = orb.fit_transform
        elif featurizer_selected.lower() == 'sift':
            sift = SIFT()
            self.image_featurizer = sift.fit_transform

        else:
            self.image_featurizer = hub.KerasLayer(featurizer_selected,
                                              input_shape=(224, 224, 3), trainable=False)
        self.preprocessor = tf.keras.Sequential([tf.keras.layers.Resizing(224, 224),
                                            tf.keras.layers.Rescaling(1. / 255)])

    def _extract_features(self,is_train):
        """
        Computes the features for the input data using the image featurizer

        Parameters
        ----------

        is_train : boolean
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
        input_data = self.train_ds if is_train else self.val_ds
        for image_batch, labels_batch in tqdm(input_data):
            preprocessed_image_batch = self.preprocessor(image_batch)
            image_batch_features = self.image_featurizer(preprocessed_image_batch)
            image_batch_features = image_batch_features if \
            isinstance(image_batch_features,list) else image_batch_features
            if features_array is None:
                features_array = image_batch_features
            else:
                features_array = np.vstack((features_array, image_batch_features))
            if input_labels is None:
                input_labels = labels_batch
            else:
                input_labels = np.concatenate([input_labels, labels_batch.numpy()])

        if features_array is None:
            raise TypeError

        if is_train:
            np.save('train_features.npy', features_array)
            np.save('train_labels.npy', input_labels)
        else:
            np.save('test_features.npy', features_array)
            np.save('test_labels.npy', input_labels)

        features_array = features_array.squeeze()

        if features_array.ndim >= 3:
            features_array = features_array.reshape(*features_array.shape\
                [:(-features_array.ndim + 1)], -1)
        return features_array, input_labels

    @staticmethod
    def _accuracy_scores(train_features,train_labels,test_features,test_labels):
        """
        Applies KNeighborsClassifier on the input data
        and return the accuracy obtained on training and testing sets.

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

        train_accuracy : float
            The accuracy computed on training set

        test_accuracy : float
            The accuracy computed on testing set

        """

        classifier = KNeighborsClassifier(weights="distance")
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

    @staticmethod
    def _accuracy_stats(train_features,train_labels,test_features,test_labels):
        """
        Applies PCA on the data over several number of
        dimensions and return a consolidated evaluation statistics.

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
        for dimension in dimensions:
            pca = PCA(n_components=dimension)
            reduced_train_features = pca.fit_transform(train_features)
            reduced_test_features = pca.transform(test_features)
            train_accuracy, test_accuracy = ImageFeatureEvaluator._accuracy_scores\
            (reduced_train_features,train_labels,reduced_test_features,test_labels)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
        results_frame = pd.DataFrame(list(zip(dimensions, train_accuracies, test_accuracies)),
                                     columns=['Dimensions', 'Train_Accuracy', 'Test_Accuracy'])

        return results_frame

    @staticmethod
    def feature_loader_and_provide_score(main_path):
        """
        Computes the stats related to evaluation metric by using pre-computed features


        Parameters
        ----------

        main_path : string
            Provides the path to the directory that contains precomputed features and labels.
            The directory should contains numpy files with names 'train_features','train_labels',
            'test_features','test_labels'.

        Returns
        -------

        results_frame : A pandas.DataFrame object
            Pandas dataframe which consists of the evaluation statistics.

        """

        train_features = np.load(main_path+'/train_features.npy')
        train_labels = np.load(main_path+'/train_labels.npy')
        test_features = np.load(main_path+'/test_features.npy')
        test_labels = np.load(main_path+'/test_labels.npy')
        results_frame = ImageFeatureEvaluator._accuracy_stats(train_features,train_labels,
                        test_features,test_labels)

        return results_frame


    def evaluate(self):
        """
          Consolidates all the below defined steps of the process of image featurizer evaluation:
          1. Calculates the features for the train and test set
          2. Computes the results dataframe containing the accuracy scores.

        Returns
        -------
        results_frame : A pandas.DataFrame object
            Pandas dataframe which consists of the evaluation statistics.

        """

        train_features,train_labels = self._extract_features(True)
        test_features, test_labels = self._extract_features(False)
        results_frame = ImageFeatureEvaluator._accuracy_stats(train_features,
                                            train_labels,test_features, test_labels)
        return results_frame


if __name__ == "__main__":
    import doctest
    print("Running doctests...")
    doctest.testmod(optionflags=doctest.ELLIPSIS)
