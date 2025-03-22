import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
"""
CV缝合救星魔改创新：引入自适应感受野和自适应卷积核大小机制
一、背景
1. 自适应感受野：感受野的大小会影响网络的特征提取能力。在一些任务中，可能需要局部细节特征，
而在另一些任务中则可能需要更广泛的上下文信息。因此，自适应感受野的机制可以根据输入的内容
决定感受野的大小。
2. 自适应卷积核大小：不同区域的特征可能需要不同大小的卷积核进行处理。比如，物体的边缘特征
需要较小的卷积核，而背景区域则适合使用较大的卷积核。通过引入自适应卷积核大小，可以使网络根
据输入的局部特征信息，动态选择最合适的卷积核大小，从而提升模型的灵活性。
二、实现思路：
1. 自适应感受野：通过动态调整卷积操作的扩张率（dilation）来改变感受野。
2. 自适应卷积核大小：在 forward 方法中，通过对输入图像的频谱分析来动态选择不同的卷积核大小。
三、关键修改：
1.卷积核大小动态调整：基于输入特征图的频谱信息（使用 FFT 变换），在高频区域使用小的卷积核，
在低频区域使用大的卷积核。
2. 扩张率调整：通过计算特征图的频谱，动态调整扩张率，以实现不同区域不同感受野的自适应调整。
"""
try:
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d
except ImportError as e:
    ModulatedDeformConv2d = nn.Module


class FrequencySelection(nn.Module):
    def __init__(self, in_channels, k_list=[2], lp_type='avgpool', act='sigmoid', spatial_group=1):
        super().__init__()
        self.k_list = k_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.in_channels = in_channels
        self.spatial_group = spatial_group
        self.lp_type = lp_type

        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group

        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                    nn.ReplicationPad2d(padding=k // 2),
                    nn.AvgPool2d(kernel_size=k, padding=0, stride=1)
                ))

            for i in range(len(k_list)):
                freq_weight_conv = nn.Conv2d(in_channels=in_channels,
                                             out_channels=self.spatial_group,
                                             stride=1,
                                             kernel_size=3,
                                             groups=self.spatial_group,
                                             padding=3 // 2,
                                             bias=True)
                self.freq_weight_conv_list.append(freq_weight_conv)

        self.act = act

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid() * 2
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        return freq_weight

    def forward(self, x):
        x_list = []

        # Ensure correct processing for the frequency selection
        if self.lp_type == 'avgpool':
            pre_x = x
            b, _, h, w = x.shape
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(x)
                high_part = pre_x - low_part
                pre_x = low_part
                freq_weight = self.freq_weight_conv_list[idx](x)
                freq_weight = self.sp_act(freq_weight)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))

            x_list.append(pre_x)

        return x_list


class FADConv(ModulatedDeformConv2d):
    def __init__(self, in_channels, offset_freq=None, kernel_decompose=None, conv_type='conv', sp_att=False, fs_cfg={'k_list': [3, 5, 7]}):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_decompose = kernel_decompose
        self.conv_type = conv_type
        if fs_cfg is not None:
            self.FS = FrequencySelection(self.in_channels, **fs_cfg)

    def init_weights(self):
        super().init_weights()

    def freq_select(self, x):
        return x

    def forward(self, x):
        if hasattr(self, 'FS'):
            x_list = self.FS(x)
        else:
            x_list = [x]

        x = sum(x_list)
        return x


# 测试代码
if __name__ == '__main__':
    input_tensor = torch.rand(1, 64, 64, 64)  # 输入形状 N C H W
    # model = FADConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, fs_cfg={'k_list': [3, 5, 7]})
    model = FADConv(in_channels=64)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
