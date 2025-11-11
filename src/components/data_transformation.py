import os
import sys
from logger import logging
from exception import CustomException
import pandas as pd
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
from utils import save_obj

class DataTransformationConfig:
    preprocesor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_cofig = DataTransformationConfig()


    def get_data_transformer_obj(self):
        """
        This function is responsible for data transformation.
        """
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False)),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('one hot encoder', OneHotEncoder()),
                ]
            ) 


            logging.info("categorical columns encoding completed.")
            logging.info("numerical columns scaling completed.")


            logging.info(f"categorical columns: {categorical_columns}")
            logging.info(f"numerical_columns: {numerical_columns}")

            preprocessor = ColumnTransformer(

                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("the train and test data has been read.")
            logging.info("obtaining preprocessor object.")

            preprocessor_obj = self.get_data_transformer_obj()

            target_column_name = "math_score"
            numerical_columns = ['writing_score','reading_score']
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing obj on train df and test df.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saved preprocessing object.")

            save_obj(
                file_path= self.data_transformation_cofig.preprocesor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_cofig.preprocesor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e, sys)