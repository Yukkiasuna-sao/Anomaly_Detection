import pandas as pd
from tensorflow.keras.utils import get_file
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def getData(file_path):
    try:
        path = get_file(file_path, origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')

    except:
        print('Error Downloading')
        raise

    df = pd.read_csv(path, header = None)
    df.dropna(inplace = True, axis = 1)

    
    df.columns = [
        'duration',
        'protocol_type',
        'service',
        'flag',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations',
        'num_shells',
        'num_access_files',
        'num_outbound_cmds',
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'outcome'
    ]
    return df


def Preprocessing(df, cat_col_idx):
    df_columns = df.columns.tolist()
        
    numerical_columns = np.delete(df_columns, cat_col_idx)
   
    std = StandardScaler()
    
    for col in numerical_columns:
        df[col] = std.fit_transform(df[[col]])
    
    def encode_text_dummy(df, name):
        dummies = pd.get_dummies(df[name])
        
        for x in dummies.columns:
            dummy_name = "{}-{}".format(name, x)
            df[dummy_name] = dummies[x]
        
        df.drop(name, axis = 1, inplace = True)
    
    encode_text_dummy(df, 'protocol_type')
    encode_text_dummy(df, 'service')
    encode_text_dummy(df, 'flag')
    encode_text_dummy(df, 'logged_in')
    encode_text_dummy(df, 'is_host_login')
    encode_text_dummy(df, 'is_guest_login')
    
    df.dropna(inplace = True, axis = 1)
    
    return df
        
def SplitData(df, testsize = None, seed = None):
    if testsize == None:
        raise AssertionError("Testsize must be defined.")
    normal = df['outcome'] == 'normal.'
    attack = df['outcome'] != 'normal.'
    
    df.drop(columns = 'outcome', inplace = True)
    
    df_normal = df[normal]
    df_attack = df[attack]
    
    x_normal = df_normal.values
    x_attack = df_attack.values
    
    x_normal_train, x_normal_test = train_test_split(x_normal, test_size = testsize, random_state = seed)
    
    return x_normal_train, x_normal_test, x_attack