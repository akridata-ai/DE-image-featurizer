"""
evaluate.py
Provides the functinality to compare different image featurizers
Creates a featurizer from tf-hub or traditional repositoy based models.
Applies PCA to reduce feature dimensions.
Provides KNN accuracy scores for different set of dimensions
"""
# pylint: disable=wrong-import-position
import sys
sys.path.append("")
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from featurize.descriptor import HOG, LBP, ORB, SIFT


# Different number of components used during PCA.
DEF_N_COMPONENTS = [2, 8, 16, 32, 64, 128]
# The dictionary describing different traditional CV featurizer.
FEATURIZER_DICT = {'hog': HOG, 'lbp': LBP, 'orb': ORB, 'sift': SIFT}
# Random seed for shuffling and transformations
SEED = 123


class ImageFeatureEvaluator:
    """
    Provides the functionality to compare different image featurizers using KNN accuracy scores.
    1. The dataset is divided into train and test sets.
    2. Features are computed using the featurizer provided.
    3. Comparative analysis is shown on different numbers of reduced dimensions using PCA.

    Parameters
    -------------
    data_dir: string
        The path to the image dataset

    featurizer: string, default=hog
        Name of a traditional CV featurizer, or URL to a tf-hub featurizer model.
        Accepted values for names are 'hog', 'lbp', 'orb', 'sift'.

    batch_size: int, default=32
        The size of the batch of the data

    splits: boolean, default=False
        Whether the directory is further divided into train and test folders.

    num_features : int
            Set the number of features to be extracted for each image
            (applicable to ORB and SIFT featurizers).

    save_features: boolean
        Whether to save the extracted features in present working directory.
    fastThreshold : int
            Determine the FAST threshold(used in case of ORB featurizer)
    edgeThreshold : int
            Determine the harris corner threshold(used in case of orb featurizer.)

    Attributes
    ----------
    batch_size: integer
        Size of batches of data.

    validation_split: float, default=0.2
        The fraction of data that is to be used as validation data.

    splits: boolean
        Whether the directory is further divided into train and test folders.

    input_shape: tuple of shape(height,width)
        The dimensions of image required by the image featurizer.

    train_ds:  A `tf.data.Dataset` object
        The training dataset.

    traditional: boolean
        Whether the featurizer is traditional or tf_hub based

    val_ds:  A `tf.data.Dataset` object
        The testing dataset.

    save_features: boolean
        Whether to save the extracted features in present working directory.

    featurizer: A `tensorflow_hub.keras_layer` object or a traditional CV
                featurizer object.
        The image featurizer object to be used for extracting features.

    preprocessor: A `keras.engine.sequential` object
        The basic preprocessor object required to standardise input.

    Examples
    ----------
    Select the path to the dataset where the training and
    testing set are divided as given below
    if splits==False

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

    Let's try on a simple tensorflow dataset.We will use flowers dataset.
    >>> import pathlib
    >>> DATASET_URL = "https://storage.googleapis.com/" \
    "download.tensorflow.org/example_images/flower_photos.tgz"
    >>> archive = tf.keras.utils.get_file(origin=DATASET_URL, extract=True)
    ...
    >>> data_dir = pathlib.Path(archive).with_suffix('')

    We will use a model from tensorflow hub.
    The class will download the model using the url.
    >>> MODEL_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5'

    The input shape dimensions according to the featurizer being used.
    >>> input_shape = (96,96)
    >>> evaluator = ImageFeatureEvaluator(data_dir,'orb',20,input_shape,0.2,False,edgeThreshold=15,fastThreshold=15)
    Found...

    The results will be returned as a dataframe
    containing different feature dimensions and the training
    and testing accuracies corresponding to them.
    >>> results = evaluator.evaluate()

     #It has DEF_N_COMPONENTS number of rows
    >>> results.shape
    (6, 3)

    List of columns present in the resultant dataframe
    >>> np.sort(results.columns)
    array(['Dimensions', 'Test_Accuracy', 'Train_Accuracy'], dtype=object)
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self, data_dir, featurizer='hog', batch_size=32, input_shape=(224, 224),
                 validation_split=0.2, splits=False,num_features=32, save_features=False,edgeThreshold=20,fastThreshold=20):
        """
        Constructor for the ImageFeatureEvaluator class
        """
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.validation_split = validation_split
        self.splits = splits
        self._load_data(data_dir)
        self._load_featurizer(featurizer,num_features,edgeThreshold,fastThreshold)
        self.save_features = save_features

    def _load_data(self, data_dir):
        """
        Creates training and testing dataset from a directory
        of images where labels are inferred from directory structure

        Parameters
        -------------
        data_dir: string
            The path to the directory containing the image dataset.
        """
        if self.splits:
            self.train_ds = tf.keras.utils.image_dataset_from_directory(data_dir + '/train',
                                                    seed=SEED, batch_size=self.batch_size)
            self.val_ds = tf.keras.utils.image_dataset_from_directory(data_dir + '/test',
                                                    seed=SEED, batch_size=self.batch_size)
        else:
            self.train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                    validation_split=self.validation_split,
                                                    subset='training',
                                                    seed=SEED, batch_size=self.batch_size)
            self.val_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                    validation_split=self.validation_split,
                                                    subset='validation',
                                                    seed=SEED, batch_size=self.batch_size)

    def _load_featurizer(self, featurizer,num_features,edgeThreshold,fastThreshold):
        """
        Instantiates the model from TF-hub library or
        from the featurizers present in the DE-image-featurizer repository
        and also creates a preprocessor to standardise the input.

        Parameters
        -------------
        featurizer: string
            Name of a traditional CV featurizer, or URL to a tf-hub featurizer model.
            Accepted values for names are 'hog', 'lbp', 'orb' and 'sift'.

        num_features : int
            Set the number of features to be extracted for each image
            (applicable to ORB and SIFT featurizers).
        """
        if featurizer.lower() in FEATURIZER_DICT:
            self.traditional = True
            if featurizer.lower()=='sift':
                self.featurizer = FEATURIZER_DICT[featurizer.lower()](num_features=num_features).fit_transform
            elif featurizer.lower()=='orb':
                self.featurizer = FEATURIZER_DICT[featurizer.lower()](num_features=num_features,edgeThreshold=edgeThreshold,fastThreshold=fastThreshold).fit_transform
            else:
                self.featurizer = FEATURIZER_DICT[featurizer.lower()]().fit_transform
        else:
            self.traditional = False
            self.featurizer = hub.KerasLayer(featurizer,
                                             input_shape=(*self.input_shape,3), trainable=False)
        self.preprocessor = tf.keras.Sequential([tf.keras.layers.Resizing(*self.input_shape),
                                                 tf.keras.layers.Rescaling(1. / 255)])

    def _extract_features(self, is_train):
        """
        Computes the features for the input data using the image featurizer.

        Parameters
        ----------
        is_train: boolean
            To distinguish between training and testing set

        Returns
        -------
        features_array: numpy.ndarray of shape (train_data_size,num_of_features)
            Numpy array containing extracted features

        input_labels: numpy.ndarray of shape (train_data_size,)
            Numpy array containing corresponding labels

        """
        features_array = []
        input_labels = []
        input_data = self.train_ds if is_train else self.val_ds
        for image_batch, labels_batch in tqdm(input_data):
            if self.traditional:
                preprocessed_image_batch = image_batch.numpy().astype(np.uint8)
            else:
                preprocessed_image_batch = self.preprocessor(image_batch)
            image_batch_features = self.featurizer(preprocessed_image_batch)
            features_array.extend(image_batch_features)
            input_labels.extend(labels_batch)
        features_array = np.asarray(features_array)
        input_labels = np.asarray(input_labels)
        features_array = features_array.reshape(*features_array.
                                                    shape[:(-1 * features_array.ndim + 1)], -1)
        if self.save_features:
            if is_train:
                np.save('train_features.npy', features_array)
                np.save('train_labels.npy', input_labels)
            else:
                np.save('test_features.npy', features_array)
                np.save('test_labels.npy', input_labels)
        return features_array, input_labels

    @staticmethod
    def _accuracy_scores(train_features, train_labels, test_features, test_labels):
        """
        Applies KNeighborsClassifier on the input data
        and return the accuracy obtained on training and testing sets.

        Parameters
        ----------
        train_features: numpy.ndarray of shape (train_data_size,num_of_features)
                The extracted features for training data

        train_labels: numpy.ndarray of shape (train_data_size,)
                The labels for training data

        test_features: numpy.ndarray of shape (test_data_size,num_of_features)
                The extracted features for testing data

        test_labels: numpy.ndarray of shape (test_data_size,)
                The labels for testing data

        Returns
        -------
        train_accuracy: float
            The accuracy computed on training set

        test_accuracy: float
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
    def _accuracy_stats(train_features, train_labels, test_features, test_labels):
        """
        Applies PCA on the data over several number of
        dimensions and return a consolidated evaluation statistics.

        Parameters
        ----------
        train_features: numpy.ndarray of shape (train_data_size,num_of_features)
            The extracted features for training data

        train_labels: numpy.ndarray of shape (train_data_size,)
            The labels for training data

        test_features: numpy.ndarray of shape (test_data_size,num_of_features)
            The extracted features for testing data

        test_labels: numpy.ndarray of shape (test_data_size,)
            The labels for testing data

        Returns
        -------
        results_frame: A pandas.DataFrame object
            Pandas dataframe which consists of the evaluation statistics.
        """
        DEF_N_COMPONENTS.sort()
        max_dim = DEF_N_COMPONENTS[-1]
        train_accuracies = []
        test_accuracies = []
        pca = PCA(n_components=max_dim)
        reduced_train_features = pca.fit_transform(train_features)
        reduced_test_features = pca.transform(test_features)
        for dimension in DEF_N_COMPONENTS:
            train_accuracy, test_accuracy = ImageFeatureEvaluator._accuracy_scores \
                (reduced_train_features[:, :dimension], train_labels,
                 reduced_test_features[:, :dimension], test_labels)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
        results_frame = pd.DataFrame(list(zip(DEF_N_COMPONENTS, train_accuracies, test_accuracies)),
                                     columns=['Dimensions', 'Train_Accuracy', 'Test_Accuracy'])
        return results_frame

    @staticmethod
    def feature_loader_and_provide_score(main_path):
        """
        Computes the stats related to evaluation metric by using pre-computed features
        Provide the paths to the extracted train and test data features from a suitable
        featurizer with their corresponding labels and it will return the evaluation
        statistics.

        Parameters
        ----------
        main_path: string
            Provides the path to the directory that contains precomputed features and labels.
            The directory should contains numpy files with names 'train_features','train_labels',
            'test_features','test_labels'.

        Returns
        -------
        results_frame: A pandas.DataFrame object
            Pandas dataframe which consists of the evaluation statistics.

        """
        train_features = np.load(main_path + '/train_features.npy')
        train_labels = np.load(main_path + '/train_labels.npy')
        test_features = np.load(main_path + '/test_features.npy')
        test_labels = np.load(main_path + '/test_labels.npy')
        results_frame = ImageFeatureEvaluator._accuracy_stats(train_features, train_labels,
                                                              test_features, test_labels)
        return results_frame

    def evaluate(self):
        """
          Consolidates all the below defined steps of the process of image featurizer evaluation:
          1. Calculates the features for the train and test set
          2. Computes the results dataframe containing the accuracy scores.

        Returns
        -------
        results_frame: A pandas.DataFrame object
            Pandas dataframe which consists of the evaluation statistics.
        """
        train_features, train_labels = self._extract_features(True)
        test_features, test_labels = self._extract_features(False)
        results_frame = ImageFeatureEvaluator._accuracy_stats(train_features,
                                        train_labels, test_features, test_labels)
        return results_frame


if __name__ == "__main__":
    import doctest

    print("Running doctests...")
    doctest.testmod(optionflags=doctest.ELLIPSIS)
