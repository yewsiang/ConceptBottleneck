
import os

MODEL_NAME = 'resnet18'
N_DATALOADER_WORKERS = 2
CACHE_LIMIT = 1e8
EST_TIME_PER_EXP = 4
BASE_DIR = '/juice/u/yewsiang/code/'
DATA_DIR = '/juice/scr/oai/'
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
IMG_DIR_PATH = os.path.join(DATA_DIR, 'proc_npz_0.5/%s/show_both_knees_True_downsample_factor_None_normalization_method_our_statistics')
IMG_CODES_FILENAME = 'image_codes.pkl'
NON_IMG_DATA_FILENAME = 'non_image_data.csv'

Y_COLS = ['xrkl']
CLINICAL_CONTROL_COLUMNS = ['xrosfm', 'xrscfm', 'xrcyfm', 'xrjsm', 'xrchm', 'xrostm', 'xrsctm', 'xrcytm', 'xrattm',
                            'xrkl', 'xrosfl', 'xrscfl', 'xrcyfl', 'xrjsl', 'xrchl', 'xrostl', 'xrsctl', 'xrcytl',
                            'xrattl']
CLASSES_PER_COLUMN = {'xrosfm': 4, 'xrscfm': 4, 'xrcyfm': 2, 'xrjsm': 4, 'xrchm': 2, 'xrostm': 4, 'xrsctm': 4,
                      'xrcytm': 3, 'xrattm': 4, 'xrkl': 4, 'xrosfl': 4, 'xrscfl': 4, 'xrcyfl': 2, 'xrjsl': 4,
                      'xrchl': 2, 'xrostl': 4, 'xrsctl': 4, 'xrcytl': 2, 'xrattl': 3 }
CONCEPTS_WO_KLG = ['xrosfm', 'xrscfm', 'xrcyfm', 'xrjsm', 'xrchm', 'xrostm', 'xrsctm', 'xrcytm', 'xrattm',
                   'xrosfl', 'xrscfl', 'xrcyfl', 'xrjsl', 'xrchl', 'xrostl', 'xrsctl', 'xrcytl', 'xrattl']
CONCEPTS_UNBALANCED = ['xrcyfm', 'xrchm', 'xrcytm', 'xrattm', 'xrcyfl', 'xrchl', 'xrcytl', 'xrattl']
CONCEPTS_BALANCED = ['xrosfm', 'xrscfm', 'xrjsm', 'xrostm', 'xrsctm', 'xrosfl', 'xrscfl', 'xrjsl', 'xrostl', 'xrsctl']

# Mu and sigma for A and y of train dataset
TRANSFORM_STATISTICS_TRAIN = {
    'xrosfm': {'mu': 0.5574507966260543, 'std': 0.982402102580231},
    'xrscfm': {'mu': 0.3287722586691659, 'std': 0.7063055371065486},
    'xrjsm': {'mu': 0.546813495782568, 'std': 0.8276194752799476},
    'xrostm': {'mu': 0.5544985941893158, 'std': 0.8192931719509258},
    'xrsctm': {'mu': 0.3501874414245548, 'std': 0.7277656712272161},
    'xrosfl': {'mu': 0.4111996251171509, 'std': 0.822236627918042},
    'xrscfl': {'mu': 0.08692596063730085, 'std': 0.39926782555526946},
    'xrjsl': {'mu': 0.13552014995313966, 'std': 0.5023994969406755},
    'xrostl': {'mu': 0.3611059044048735, 'std': 0.7567221565202209},
    'xrsctl': {'mu': 0.08406747891283974, 'std': 0.3928446987641911}
}
