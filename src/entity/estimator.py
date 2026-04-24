import sys
from src.exception import MyException


class MyModel:
    def __init__(self, preprocessing_object, trained_model_object):
        try:
            self.preprocessing_object = preprocessing_object
            self.trained_model_object = trained_model_object
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, X):
        try:
            X_transformed = self.preprocessing_object.transform(X)
            return self.trained_model_object.predict(X_transformed)
        except Exception as e:
            raise MyException(e, sys)