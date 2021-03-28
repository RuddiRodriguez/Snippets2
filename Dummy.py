"""
Script to estimated a base line of the model
based on a dummy classifier 
"""


# Load libraries
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split


def dummy_classifier(features, target):
    # Split into training and test set
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, random_state=0)
    # Create dummy classifier
    dummy = DummyClassifier(strategy='uniform', random_state=1)
    # "Train" model
    dummy.fit(features_train, target_train)
    # Get accuracy score
    print (dummy.score(features_test, target_test))
    return dummy.score(features_test, target_test)


