import torch
import torch.nn as nn

class MSMDDNet(nn.Module):
    def __init__(self, in_ch=32):
        """
        MS-MDDNet Baseline Architecture
        Default in_ch=32 matches the paper's exact 34,812 parameter count.
        """
        super(MSMDDNet, self).__init__()
        
        # 1. F_pre: Spatial Information Integration[cite: 2]
        half_ch = in_ch // 2
        # Fuses left and right hemisphere channels independently[cite: 2]
        self.sconv_left = nn.Conv1d(in_channels=half_ch, out_channels=1, kernel_size=1)
        self.sconv_right = nn.Conv1d(in_channels=in_ch - half_ch, out_channels=1, kernel_size=1)
        
        # 2. F_S: Spatial Analysis[cite: 2]
        # Integrates the 2 hemisphere channels into 8 temporal feature channels[cite: 2]
        self.sconv_spatial = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=1)
        self.gn_spatial = nn.GroupNorm(num_groups=2, num_channels=8) 
        self.relu = nn.ReLU() # Paper's ablation study confirms ReLU outperforms Swish/Tanh[cite: 2]
        
        # 3. F_MS: Multiscale Analysis[cite: 2]
        # Three depth-wise temporal convolutions at scales SR/2 (125), SR/4 (62), and SR/8 (31)[cite: 2]
        # Depth multiplier is 2, so 8 input groups -> 16 output channels[cite: 2]
        self.dtconv1 = nn.Conv1d(8, 16, kernel_size=125, groups=8, padding='same')
        self.bn1 = nn.BatchNorm1d(16)
        
        self.dtconv2 = nn.Conv1d(8, 16, kernel_size=62, groups=8, padding='same')
        self.bn2 = nn.BatchNorm1d(16)
        
        self.dtconv3 = nn.Conv1d(8, 16, kernel_size=31, groups=8, padding='same')
        self.bn3 = nn.BatchNorm1d(16)
        
        # Average pooling reduces the temporal dimension[cite: 2]
        self.avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)
        
        # 4. F_HFL: High-level Feature Learning[cite: 2]
        # Input is 48 channels from the concatenated multiscale block[cite: 2]
        # First Separable Block[cite: 2]
        self.hfl_dtconv1 = nn.Conv1d(48, 96, kernel_size=31, groups=48, padding='same')
        self.hfl_bn1 = nn.BatchNorm1d(96)
        self.hfl_pconv1 = nn.Conv1d(96, 96, kernel_size=1)
        self.hfl_bn2 = nn.BatchNorm1d(96)
        
        # Second Separable Block[cite: 2]
        self.hfl_dtconv2 = nn.Conv1d(96, 192, kernel_size=31, groups=96, padding='same')
        self.hfl_bn3 = nn.BatchNorm1d(192)
        self.hfl_pconv2 = nn.Conv1d(192, 48, kernel_size=1)
        self.hfl_bn4 = nn.BatchNorm1d(48)
        
        # Shortcut Connection matching output dimensions without BN[cite: 2]
        self.skip_pconv = nn.Conv1d(48, 48, kernel_size=1)
        
        # Global Average Pooling prevents overfitting on the final features[cite: 2]
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 5. F_C: Classifier[cite: 2]
        self.fc = nn.Linear(48, 2)
        
    def forward(self, x):
        # 1. Spatial Information Integration
        half_ch = x.shape[1] // 2
        x_left = x[:, :half_ch, :]
        x_right = x[:, half_ch:, :]
        
        out_left = self.sconv_left(x_left)
        out_right = self.sconv_right(x_right)
        x_pre = torch.cat((out_left, out_right), dim=1) 
        
        # 2. Spatial Analysis
        x_s = self.sconv_spatial(x_pre)
        x_s = self.gn_spatial(x_s)
        x_s = self.relu(x_s) 
        
        # 3. Multiscale Analysis
        ms1 = self.relu(self.bn1(self.dtconv1(x_s)))
        ms2 = self.relu(self.bn2(self.dtconv2(x_s)))
        ms3 = self.relu(self.bn3(self.dtconv3(x_s)))
        
        x_ms = torch.cat((ms1, ms2, ms3), dim=1) 
        x_ms = self.avg_pool(x_ms) 
        
        # 4. High-level Feature Learning
        hfl1 = self.relu(self.hfl_bn1(self.hfl_dtconv1(x_ms)))
        hfl1 = self.relu(self.hfl_bn2(self.hfl_pconv1(hfl1)))
        
        hfl2 = self.relu(self.hfl_bn3(self.hfl_dtconv2(hfl1)))
        hfl2 = self.relu(self.hfl_bn4(self.hfl_pconv2(hfl2)))
        
        skip = self.skip_pconv(x_ms)
        
        x_hfl = hfl2 + skip 
        x_hfl = self.global_avg_pool(x_hfl).squeeze(-1) 
        
        # 5. Classifier
        logits = self.fc(x_hfl)
        return logits

def test_param_count():
    model = MSMDDNet(in_ch=32)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {params}")
    assert params == 34812, f"Failed: Expected 34812, got {params}"
    print("Success: Parameter count exactly matches the MS-MDDNet paper (34,812).")

if __name__ == "__main__":
    test_param_count()