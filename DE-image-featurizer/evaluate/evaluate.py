class ImageFeatureEvaluator:
    """
    Evaluation of featurizer

    Methods
    -------

    load_data
        Create a training and validation dataset given the directory of image dataset with labels inferred in the directory structure

    evaluate
        Computes the features according to the featuriser provided and returns the stats related to evaluation metric used

    feature_loader_and_provide_score
        Computes the stats related to evaluation metric by using pre computed features

    load_featurizer
        Instantiate the featuriser from tf-hub

    extract_features
        Method to extract the features from the input data using the featurizer

    accuracy_score
        Method to calculate the evaluation score provided the training and testing data

    accuracy_stats
        Provides consolidated results by evaluating the featurizer on different set of reduced dimensions
    """

    def load_data(data_dir):
        """
         Creates training and validation dataset from a directory of images where labels are inferred from directory structure

        Parameters
        ----------
        data_dir : string
            Provides the path to parent directory where data is present

        Returns
        -------
        train_ds
            The training dataset

        test_ds
            The testing dataset
        """
        train_ds = tf.keras.utils.image_dataset_from_directory(data_dir + '/train',
                                                               seed=123, batch_size=32)
        val_ds = tf.keras.utils.image_dataset_from_directory(data_dir + '/test',
                                                             seed=123, batch_size=32)

        return train_ds, val_ds

    def evaluate(data_dir, featurizerSelected, saving_path=""):
        """
          Consolidates all the steps of the process of image featurizer evaluation

        Parameters
        ----------
        data_dir : string
            Provides the path to parent directory where data is present
        featurizerSelected : string
            Hyperlink to the Image featurizer module
        saving_path : string
            Provides the path to where the extracted features are to be saved

        Returns
        -------
        results_frame
            Pandas dataframe which consists of the evaluation statistics.

        """
        train_ds, val_ds = ImageFeatureEvaluator.load_data(data_dir)
        featurizer, preprocessor = ImageFeatureEvaluator.load_featurizer(featurizerSelected)
        train_features, train_labels = ImageFeatureEvaluator.extract_features(featurizer,
                                                                             preprocessor, train_ds,
                                                                             saving_path, True)
        test_features, test_labels = ImageFeatureEvaluator.extract_features(featurizer,
                                                                           preprocessor, val_ds,
                                                                           saving_path, False)
        results_frame = ImageFeatureEvaluator.accuracy_stats(train_features, test_features,
                                                            train_labels, test_labels)

        return results_frame

    def feature_loader_and_provide_score(train_features_path, train_labels_path,
                                         test_features_path, test_labels_path):
        """
        Computes the stats related to evaluation metric by using pre computed features


        Parameters
        ----------
        train_features_path : string
            Provides the path to the precomputed features for training data.
        train_labels_path : string
            Provides the path to the labels associated to training data.
        test_features_path : string
            Provides the path to the precomputed features for the testing data.
        test_labels_path : string
            Provides the path to the labels associated to testing data.

        Returns
        -------
        results_frame
            Pandas dataframe which consists of the evaluation statistics.

        """
        train_features = np.load(train_features_path)
        train_labels = np.load(train_labels_path)
        test_features = np.load(test_features_path)
        test_labels = np.load(test_labels_path)
        results_frame = ImageFeatureEvaluator.accuracy_stats(train_features, test_features,
                                                            train_labels, test_labels)

        return results_frame

    def load_featurizer(featurizerSelected):
        """
        Instantiates the model from TF-hub library and also creates a preprocessor to standardise the input.

        Parameters
        ----------
        featurizerSelected : string
            Hyperlink to the Image featurizer module

        Returns
        -------
        image_featurizer
            The image featurizer model to be used for extracting features.

        preprocessor
            The basic preprocessor function required to standardise input
        """
        image_featurizer = hub.KerasLayer(featurizerSelected,
                                          input_shape=(224, 224, 3), trainable=False)
        preprocessor = tf.keras.Sequential([layers.Resizing(224, 224),
                                            layers.Rescaling(1. / 255)])

        return image_featurizer, preprocessor

    def extract_features(image_featurizer, preprocessor, input_data, saving_path, train):
        """
        Computes the features for the input data using the image featurizer

        Parameters
        ----------
        image_featurizer : function_like
            The image featurizer model imported from TF-hub
        preprocessor : function_like
        The preprocessing  function comprising of rescaling and resizing.
        input_data : tensor_like
            The data containing images as elements
        saving_path : string
            The path to save extracted features
        train : boolean
            To distinguish between training and testing set

        Returns
        -------
        features_array
            Numpy array containing extracted features
        input_labels
            Numpy array containing corresponding labels

        """
        features_array = None
        input_labels = None

        for image_batch, labels_batch in tqdm(input_data):
            image_batch = preprocessor(image_batch)
            image_batch = image_featurizer(image_batch)
            image_batch = image_batch if isinstance(image_batch, np.ndarray) else image_batch.numpy()
            if features_array is None:
                features_array = image_batch
            else:
                features_array = np.vstack((features_array, image_batch))
            if input_labels is None:
                input_labels = labels_batch
            else:
                input_labels = np.concatenate([input_labels, labels_batch.numpy()])
        if saving_path != '':
            if train:
                np.save(saving_path + '/train_features.npy', features_array)
                np.save(saving_path + '/train_labels.npy', input_labels)
            else:
                np.save(saving_path + '/test_features.npy', features_array)
                np.save(saving_path + '/test_labels.npy', input_labels)

        return features_array, input_labels

    def accuracy_score(train_features, train_labels,
                       test_features, test_labels,
                       n_neighbors):
        """
        Applies KNeighborsClassifier on the input data and return the accuracy obtained on training and testing sets.

        Parameters
        ----------
        train_features : array_like
            The extracted features for training data
        train_labels : array_like
            The labels for training data
        test_features : array_like
            The extracted features for testing data
        test_labels : array_like
            The labels for testing data
        n_neighbors : array_like
            Number of neighbors to consider for classification

        Returns
        -------
        train_accuracy
            The accuracy computed on training set
        test_accuracy
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

    def accuracy_stats(train_features, test_features,
                       train_labels, test_labels):
        """
        Applies PCA on the data over several number of dimensions and return a consolidated evaluation statistics.

        Parameters
        ----------
        train_features : array_like
            The extracted features for training data
        train_labels : array_like
            The labels for training data
        test_features : array_like
            The extracted features for testing data
        test_labels : array_like
            The labels for testing data

        Returns
        -------
        df
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
            train_accuracy, test_accuracy = ImageFeatureEvaluator.accuracy_score(reduced_train_features, train_labels,
                                                                                reduced_test_features, test_labels, 5)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
        results_frame = pd.DataFrame(list(zip(dimensions, train_accuracies, test_accuracies)),
                                     columns=['Dimensions', 'Train_Accuracy', 'Test_Accuracy'])

        return results_frame