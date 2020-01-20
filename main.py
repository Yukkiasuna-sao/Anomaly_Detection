import sys
sys.path.append('./code')
import warnings
warnings.filterwarnings('ignore')

from DataPreprocessing import getData, Preprocessing, SplitData
from Autoencoder import SimpleUncompleteAutoencoder, SimpleStackedAutoencoder, SimpleDenosingAutoencoder
from IsolationForest import SimpleIsolationForest
from OneClassSVM import SimpleOneClassSVM

""" Novelty Detection """
df = getData('kddcup.data_10_percent.gz')
df = Preprocessing(df, [1,2,3,6,11,20,21,41])

x_normal_train, x_normal_test, x_attack = SplitData(df,testsize = 0.25, seed = 42, method = "Novelty")

SUA = SimpleUncompleteAutoencoder(x_normal_train)
SUA.Modeling(x_normal_train, 25, batchsize = 50, validation_size = 0.1)
SUA.Prediction(x_normal_train, data_type = 'Insample')
SUA.Prediction(x_normal_test, data_type = 'OutOfSample')
prediction_SUA = SUA.Prediction(x_attack, data_type = 'Attack')


SSA = SimpleStackedAutoencoder(x_normal_train)
SSA.Modeling(x_normal_train, hidden_dim = 25, coding_dim = 3, batchsize = 50, validation_size = 0.1)
SSA.Prediction(x_normal_train, data_type = 'Insample')
SSA.Prediction(x_normal_test, data_type = 'OutOfSample')
prediction_SSA = SSA.Prediction(x_attack, data_type = 'Attack')


SDA = SimpleDenosingAutoencoder(x_normal_train)
SDA.Modeling(x_normal_train, hidden_dim = 25, coding_dim = 3, batchsize = 50, validation_size = 0.1, denosing_type = 'Dropout')
SDA.Prediction(x_normal_train, data_type = 'Insample')
SDA.Prediction(x_normal_test, data_type = 'OutOfSample')
prediction_SDA = SDA.Prediction(x_attack, data_type = 'Attack')

SDA = SimpleDenosingAutoencoder(x_normal_train)
SDA.Modeling(x_normal_train, hidden_dim = 25, coding_dim = 3, batchsize = 50, validation_size = 0.1, denosing_type = 'Gaussian', std = 0.1)
SDA.Prediction(x_normal_train, data_type = 'Insample')
SDA.Prediction(x_normal_test, data_type = 'OutOfSample')
prediction_SDA = SDA.Prediction(x_attack, data_type = 'Attack')

OCS = SimpleOneClassSVM(x_normal_train)
OCS.Modeling(x_normal_train, seed = 42)
OCS.Prediction(x_normal_train, data_type = 'Insample')
OCS.Prediction(x_normal_test, data_type = 'OutOfSample')
Prediction_OCS = OCS.Prediction(x_attack, data_type = 'Attack')

""" Outlier Detection """

x_train, x_test, y_train, y_test = SplitData(df, testsize = 0.25, seed = 42, method = "Outlier")
ISF = SimpleIsolationForest(x_train)
ISF.Modeling(x_train, seed = 42)
ISF.Prediction(x_train, data_type = 'Insample')
Prediction_ISF = ISF.Prediction(x_test, data_type = 'OutOfSample')
