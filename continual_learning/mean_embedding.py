import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rff_param_tuple = namedtuple('rff_params', ['w', 'b'])


##############################################################
##  distance
##############################################################
def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X ** 2, 1)
    sy = np.sum(Y ** 2, 1)
    D2 = sx[:, np.newaxis] - 2.0 * X.dot(Y.T) + sy[np.newaxis, :]
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D


def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and
        there are more slightly more 0 than 1. In this case, the m

    Return
    ------
    median distance
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)


def one_hot_embedding_torch(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


def one_hot_embedding_np(targets, num_classes=10):
    res = np.eye(num_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [num_classes])


##############################################################
##  kernels to use
##############################################################
""" we use the random fourier feature representation for Gaussian kernel """


def RFF_Gauss(n_features, X, W):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""

    W = torch.Tensor(W).to(device)
    X = X.to(device)

    XWT = torch.mm(X, torch.t(W)).to(device)
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2), 1) * torch.sqrt(2.0 / torch.Tensor([n_features])).to(device)
    return Z


""" we use a weighted polynomial kernel for labels """


def Feature_labels(labels, weights):
    # weights = class_ratios   (shape: [#classes,])
    # labels are one-hot encoding
    # This function returns the 1/class_fraction for the gt-class and 0 otherwise

    weights = weights.to(device)
    labels = labels.to(device)

    weighted_labels_feature = labels / weights

    return weighted_labels_feature


def rff_sphere(x, rff_params):
    """
    this is a Pytorch version of anon's code for RFFKGauss
    Fourier transform formula from http://mathworld.wolfram.com/FourierTransformGaussian.html
    """
    w = rff_params.w
    xwt = torch.mm(x, w.t())
    z_1 = torch.cos(xwt)
    z_2 = torch.sin(xwt)
    z_cat = torch.cat((z_1, z_2), 1)
    norm_const = torch.sqrt(torch.tensor(w.shape[0]).to(torch.float32))
    z = z_cat / norm_const  # w.shape[0] == n_features / 2
    return z


def weights_sphere(d_rff, d_enc, sig, device):
    w_freq = torch.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(sig)).to(torch.float32).to(device)
    return rff_param_tuple(w=w_freq, b=None)


def rff_rahimi_recht(x, rff_params):
    """
    implementation more faithful to rahimi+recht paper
    """
    w = rff_params.w
    b = rff_params.b
    xwt = torch.mm(x, w.t()) + b
    z = torch.cos(xwt)
    z = z * torch.sqrt(torch.tensor(2. / w.shape[0]).to(torch.float32))
    return z


def weights_rahimi_recht(d_rff, d_enc, sig, device):
    w_freq = torch.tensor(np.random.randn(d_rff, d_enc) / np.sqrt(sig)).to(torch.float32).to(device)
    b_freq = torch.tensor(np.random.rand(d_rff) * (2 * np.pi * sig)).to(device)
    return rff_param_tuple(w=w_freq, b=b_freq)


def data_label_embedding(data, labels, rff_params, mmd_type, labels_to_one_hot=False, num_classes=10, device=None, reduce='mean'):
    assert reduce in {'mean', 'sum'}
    if labels_to_one_hot:
        batch_size = data.shape[0]
        one_hots = torch.zeros(batch_size, num_classes, device=device)
        one_hots.scatter_(1, labels[:, None], 1)
        labels = one_hots

    data_embedding = rff_sphere(data, rff_params) if mmd_type == 'sphere' else rff_rahimi_recht(data, rff_params)
    embedding = torch.einsum('ki,kj->kij', [data_embedding, labels])
    return torch.mean(embedding, 0) if reduce == 'mean' else torch.sum(embedding, 0)


def noisy_dataset_embedding(train_loader, w_freq, d_rff, device, num_classes, noise_factor, mmd_type, sum_frequency=25):
    emb_acc = []
    n_data = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        bs = data.shape[0]
        data = torch.reshape(data, (bs, -1))
        emb_acc.append(data_label_embedding(data, labels, w_freq, mmd_type, labels_to_one_hot=True, num_classes=num_classes, device=device, reduce='sum'))
        n_data += data.shape[0]

        if len(emb_acc) > sum_frequency:
            emb_acc = [torch.sum(torch.stack(emb_acc), 0)]
    print('done collecting batches, n_data', n_data)
    emb_acc = torch.sum(torch.stack(emb_acc), 0) / n_data
    print(torch.norm(emb_acc), emb_acc.shape)
    noise = torch.randn(d_rff, num_classes, device=device) * (2 * noise_factor / n_data)
    noisy_emb = emb_acc + noise
    return noisy_emb


##############################################################
##  Generators
##############################################################
class Generative_Model_homogeneous_data(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(Generative_Model_homogeneous_data, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(self.bn1(hidden))
        output = self.fc2(relu)
        output = self.relu(self.bn2(output))
        output = self.fc3(output)
        return output


class Generative_Model_heterogeneous_data(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, num_categorical_inputs, num_numerical_inputs):
        super(Generative_Model_heterogeneous_data, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.num_numerical_inputs = num_numerical_inputs
        self.num_categorical_inputs = num_categorical_inputs

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(self.bn1(hidden))
        output = self.fc2(relu)
        output = self.relu(self.bn2(output))
        output = self.fc3(output)

        output_numerical = self.relu(output[:, 0:self.num_numerical_inputs])  # these numerical values are non-negative
        output_categorical = self.sigmoid(output[:, self.num_numerical_inputs:])
        output_combined = torch.cat((output_numerical, output_categorical), 1)

        return output_combined


class ConvCondGen(nn.Module):
    def __init__(self, d_code, d_hid, n_labels, nc_str, ks_str, use_sigmoid=True, batch_norm=True):
        super(ConvCondGen, self).__init__()
        self.nc = [int(k) for k in nc_str.split(',')] + [1]  # number of channels
        self.ks = [int(k) for k in ks_str.split(',')]  # kernel sizes
        d_hid = [int(k) for k in d_hid.split(',')]
        assert len(self.nc) == 3 and len(self.ks) == 2
        self.hw = 7  # image height and width before upsampling
        self.reshape_size = self.nc[0] * self.hw ** 2
        self.fc1 = nn.Linear(d_code + n_labels, d_hid[0])
        self.fc2 = nn.Linear(d_hid[0], self.reshape_size)
        self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
        self.bn2 = nn.BatchNorm1d(self.reshape_size) if batch_norm else None
        self.conv1 = nn.Conv2d(self.nc[0], self.nc[1], kernel_size=self.ks[0], stride=1, padding=(self.ks[0] - 1) // 2)
        self.conv2 = nn.Conv2d(self.nc[1], self.nc[2], kernel_size=self.ks[1], stride=1, padding=(self.ks[1] - 1) // 2)
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
        self.d_code = d_code
        self.n_labels = n_labels

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.fc2(self.relu(x))
        x = self.bn2(x) if self.bn2 is not None else x
        # print(x.shape)
        x = x.reshape(x.shape[0], self.nc[0], self.hw, self.hw)
        x = self.upsamp(x)
        x = self.relu(self.conv1(x))
        x = self.upsamp(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x

    def get_code(self, batch_size, device, return_labels=True, labels=None):
        if labels is None:  # sample labels
            labels = torch.randint(self.n_labels, (batch_size, 1), device=device)
        code = torch.randn(batch_size, self.d_code, device=device)
        gen_one_hots = torch.zeros(batch_size, self.n_labels, device=device)
        gen_one_hots.scatter_(1, labels, 1)
        code = torch.cat([code, gen_one_hots.to(torch.float32)], dim=1)
        # print(code.shape)
        if return_labels:
            return code, gen_one_hots
        else:
            return code

    def sample(self, num_samples, device):
        images = []
        labels = []
        bs = 100
        for i in range(num_samples // bs + 1):
            lab = torch.randint(self.n_labels, (bs, 1), device=device)
            code, _ = self.get_code(bs, device, labels=lab)
            img = self.forward(code)
            images.append(img)
            labels.append(lab)

        images = torch.cat(images)[:num_samples]
        labels = torch.cat(labels)[:num_samples].view(-1)
        return images, labels
