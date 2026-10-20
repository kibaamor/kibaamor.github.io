import hashlib
import os
import tarfile
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Subset

DATA_HUB = dict()
DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"

DATA_HUB["kaggle_house_train"] = (
    DATA_URL + "kaggle_house_pred_train.csv",
    "585e9cc93e70b39160e7921475f9bcd7d31219ce",
)

DATA_HUB["kaggle_house_test"] = (
    DATA_URL + "kaggle_house_pred_test.csv",
    "fa19780a7b011d9b009e8bff8e99922a8ee2eb90",
)


def download(name, cache_dir=os.path.join(".", "data")):
    """下载一个DATA_HUB中的文件，返回本地文件名。"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache

    print(f"正在从{url}下载{fname}...")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, "wb") as f:
        f.write(r.content)

    return fname


def download_extract(name, folder=None):
    """下载并解压zip/tar文件。"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, "r")
    elif ext in (".tar", ".gz"):
        fp = tarfile.open(fname, "r")
    else:
        assert False, "只有zip/tar文件可以被解压缩。"
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    """下载DATA_HUB中的所有文件。"""
    for name in DATA_HUB:
        download(name)


def preprocess_data():
    download_all()

    train_data = pd.read_csv(download("kaggle_house_train"))
    test_data = pd.read_csv(download("kaggle_house_test"))

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    all_features = pd.get_dummies(all_features, dummy_na=True)

    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)

    labels = train_data.SalePrice.values.reshape(-1, 1)
    labels_scaler = StandardScaler()
    labels_scaler.fit(labels)
    labels = labels_scaler.transform(labels)
    train_labels = torch.tensor(labels, dtype=torch.float32)

    return train_features, train_labels, test_features, labels_scaler


def cross_valid(k_fold, model, criterion, optimizer, dataset, batch_size, num_epochs):
    kf = KFold(n_splits=k_fold, shuffle=True)
    for train_indices, test_indices in kf.split(dataset):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_losses, test_losses = train(
            model, criterion, optimizer, train_dataset, batch_size, num_epochs, test_dataset=test_dataset
        )
        print("train loss", np.mean(train_losses), "test loss", np.mean(test_losses))


def train(model, criterion, optimizer, dataset, batch_size, num_epochs, *, test_dataset=None):
    train_losses, test_losses = [], []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for _ in range(num_epochs):
        model.train(True)
        for features, labels in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()

        model.train(False)
        with torch.no_grad():
            train_loss = criterion(model(dataset[:][0]), dataset[:][1])
            train_losses.append(train_loss.item())

            if test_dataset:
                test_loss = criterion(model(test_dataset[:][0]), test_dataset[:][1])
                test_losses.append(test_loss)

    return train_losses, test_losses


def infer(model, features):
    model.train(False)
    with torch.no_grad():
        return model(features)


train_features, train_labels, test_features, labels_scaler = preprocess_data()
in_features = train_features.shape[1]
out_features = 1
model = nn.Sequential(
    nn.Linear(in_features, 128),
    ReLU(),
    nn.Linear(128, 128),
    ReLU(),
    nn.Linear(128, out_features),
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataset = TensorDataset(train_features, train_labels)
num_epochs = 1000
batch_size = 64
k_fold = 5
cross_valid(k_fold, model, criterion, optimizer, dataset, batch_size, num_epochs)
