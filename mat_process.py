from scipy.io import savemat
import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from nn_predict import dataset_feature


def make_desp(dname, features, train_pids, probe_pids, gallery_pids):
    feature_mat = {
        'camID': [],
        'feature': np.array(features).transpose(),
        'ID': train_pids + probe_pids + gallery_pids,

    }
    print(feature_mat['feature'].shape)
    feature_mat['camID'].extend([1 for _ in train_pids])
    feature_mat['camID'].extend([2 for _ in probe_pids])
    feature_mat['camID'].extend([1 for _ in gallery_pids])

    savemat(dname + '_norm_feature.mat', feature_mat)

    split_mat = {
        'trials': {
            'labelsAtrain': sorted(list(set(train_pids))),
            'labelsAtest': sorted(list(set(probe_pids + gallery_pids)))
        }
    }

    savemat(dname + '_split_kcca.mat', split_mat)


def main(args):
    features, train_pids, probe_pids, gallery_pids = dataset_feature(args['ds_path'])
    make_desp('duke', features, train_pids, probe_pids, gallery_pids)


if __name__ == '__main__':
    args = {'ds_path': '/home/cwh/coding/DukeMTMC-reID'}
    main(args)
