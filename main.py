import sys
sys.path.append('./code')
import warnings
warnings.filterwarnings('ignore')

from DataPreprocessing import getData, Preprocessing, SplitData
from Autoencoder import SimpleUncompleteAutoencoder, SimpleStackedAutoencoder, SimpleDenosingAutoencoder


df = getData('kddcup.data_10_percent.gz')
df = Preprocessing(df, [1,2,3,6,11,20,21,41])

x_normal_train, x_normal_test, x_attack = SplitData(df,testsize = 0.25, seed = 42)

SUA = SimpleUncompleteAutoencoder(x_normal_train)
SUA.Modeling(x_normal_train, 25, batchsize = 50, validation_size = 0.25)
SUA.Prediction(x_normal_test, data_type = 'Insample')


SSA = SimpleStackedAutoencoder(x_normal_train)
SSA.Modeling(x_normal_train, hidden_dim = 25, coding_dim = 3, batchsize = 50, validation_size = 0.1)
SSA.Prediction(x_normal_test, data_type = 'Insample')


SDA = SimpleDenosingAutoencoder(x_normal_train)
SDA.Modeling(x_normal_train, hidden_dim = 25, coding_dim = 3, batchsize = 50, validation_size = 0.1, denosing_type = 'Dropout')
SDA.Prediction(x_normal_test, data_type = 'Insample')