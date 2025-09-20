import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder



class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.maxpool = torch.nn.AdaptiveAvgPool1d(1)


    def forward(self, x, required_features=False):
        features, trans, trans_feat, x_pre = self.feat(x)
        x = F.relu(self.bn1(self.fc1(features)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        if required_features:
            return x, features, x_pre
        else:
            return x
    
    def get_pre_global_feat(self, x):
        x = self.feat.get_pre_max_val(x)
        return x


