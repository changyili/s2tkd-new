import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import ASPP, BasicBlock, l2_normalize, make_layer
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
        )
        # freeze teacher model
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.eval()
        x1, x2, x3 = self.encoder(x)
        return (x1, x2, x3)

# class MultiFeatureFusion(nn.Module):
#     def __init__(self, channel = 64):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(channel, channel * 2, kernel_size=3, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 2),
#             nn.ReLU(),
#             nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 4),
#             nn.ReLU(),
#             nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 4),
#             nn.ReLU(),
#             nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(),
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(),
#         )

#         self.conv_ = nn.Sequential(
#             nn.Conv2d(channel * 8 * 4, channel * 8, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(),
#         )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x1, x2, x3, x4):
#         # print('x1_', x1.shape)
#         # print('x2_', x2.shape)
#         # print('x3_', x3.shape)
#         # print('x4_', x4.shape)
#         x1 = self.conv1(x1)
#         x2 = self.conv2(x2)
#         x3 = self.conv3(x3)
#         # print('x1', x1.shape)
#         # print('x2', x2.shape)
#         # print('x3', x3.shape)
#         # print('x4', x4.shape)
#         x = torch.cat([x1, x2, x3, x4], 1)
#         x = self.conv_(x)
#         return x

# class MultiFeatureFusion(nn.Module):
#     def __init__(self, channel = 64):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(channel, channel * 2, kernel_size=3, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 2),
#             nn.ReLU(),
#             # nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride = 2, padding = 1),
#             # nn.BatchNorm2d(channel * 4),
#             # nn.ReLU(),
#             # nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
#             # nn.BatchNorm2d(channel * 8),
#             # nn.ReLU(),
#         )
#         # self.conv2 = nn.Sequential(
#         #     nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride = 2, padding = 1),
#         #     nn.BatchNorm2d(channel * 4),
#         #     nn.ReLU(),
#         #     nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
#         #     nn.BatchNorm2d(channel * 8),
#         #     nn.ReLU(),
#         # )
#         # self.conv3 = nn.Sequential(
#         #     nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
#         #     nn.BatchNorm2d(channel * 8),
#         #     nn.ReLU(),
#         # )

#         self.conv_ = nn.Sequential(
#             nn.Conv2d(channel * 8 * 2, channel * 8, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(),
#         )
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(channel * 8, channel * 8, kernel_size=3, bias=False, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(),
#         )
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(channel * 8, channel * 8, kernel_size=3, bias=False, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(),
#         )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x1, x2, x3, x4):
#         x1 = self.conv1(x1)
#         x3 = F.interpolate(x3, size=x2.shape[-1], mode='bilinear', align_corners=True)
#         x4 = F.interpolate(x4, size=x2.shape[-1], mode='bilinear', align_corners=True)
#         # x2 = self.conv2(x2)
#         # x3 = self.conv3(x3)
#         # print('x1', x1.shape)
#         # print('x2', x2.shape)
#         # print('x3', x3.shape)
#         # print('x4', x4.shape)
#         x = torch.cat([x1, x2, x3, x4], 1)
#         x = self.conv_(x)
#         x = self.conv_2(self.conv_1(x))
#         return x

# class MultiFeatureFusion(nn.Module):
#     def __init__(self, channel = 64):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(channel, channel * 2, kernel_size=3, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 2),
#             nn.ReLU(),
#             nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 4),
#             nn.ReLU(),
#             # nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
#             # nn.BatchNorm2d(channel * 8),
#             # nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 4),
#             nn.ReLU(),
#         #     nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
#         #     nn.BatchNorm2d(channel * 8),
#         #     nn.ReLU(),
#         )
#         # self.conv3 = nn.Sequential(
#         #     nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
#         #     nn.BatchNorm2d(channel * 8),
#         #     nn.ReLU(),
#         # )

#         self.conv_ = nn.Sequential(
#             nn.Conv2d(channel * 20, channel * 8, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(),
#         )
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(channel * 8, channel * 8, kernel_size=3, bias=False, stride = 2, padding = 1),
#             nn.BatchNorm2d(channel * 8),
#             nn.ReLU(),
#         )
#         # self.conv_2 = nn.Sequential(
#         #     nn.Conv2d(channel * 8, channel * 8, kernel_size=3, bias=False, stride = 2, padding = 1),
#         #     nn.BatchNorm2d(channel * 8),
#         #     nn.ReLU(),
#         # )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x1, x2, x3, x4):
#         x1 = self.conv1(x1)
#         x2 = self.conv2(x2)
#         # x3 = F.interpolate(x3, size=x2.shape[-1], mode='bilinear', align_corners=True)
#         x4 = F.interpolate(x4, size=x3.shape[-1], mode='bilinear', align_corners=True)
#         # x2 = self.conv2(x2)
#         # x3 = self.conv3(x3)
#         # print('x1', x1.shape)
#         # print('x2', x2.shape)
#         # print('x3', x3.shape)
#         # print('x4', x4.shape)
#         x = torch.cat([x1, x2, x3, x4], 1)
#         x = self.conv_(x)
#         # x = self.conv_2(self.conv_1(x))
#         x = self.conv_1(x)
#         return x


class MultiFeatureFusion(nn.Module):
    def __init__(self, channel = 64):
        super().__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(channel, channel * 2, kernel_size=3, stride = 2, padding = 1),
        #     nn.BatchNorm2d(channel * 2),
        #     nn.ReLU(),
        #     nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride = 2, padding = 1),
        #     nn.BatchNorm2d(channel * 4),
        #     nn.ReLU(),
            # nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
            # nn.BatchNorm2d(channel * 8),
            # nn.ReLU(),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride = 2, padding = 1),
        #     nn.BatchNorm2d(channel * 4),
        #     nn.ReLU(),
        #     nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
        #     nn.BatchNorm2d(channel * 8),
        #     nn.ReLU(),
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride = 2, padding = 1),
        #     nn.BatchNorm2d(channel * 8),
        #     nn.ReLU(),
        # )

        self.conv_ = nn.Sequential(
            nn.Conv2d(channel * 15, channel * 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel * 8),
            nn.ReLU(),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(channel * 8, channel * 8, kernel_size=3, bias=False, stride = 2, padding = 1),
            nn.BatchNorm2d(channel * 8),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channel * 8, channel * 8, kernel_size=3, bias=False, stride = 2, padding = 1),
            nn.BatchNorm2d(channel * 8),
            nn.ReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(channel * 8, channel * 8, kernel_size=3, bias=False, stride = 2, padding = 1),
            nn.BatchNorm2d(channel * 8),
            nn.ReLU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, x3, x4):
        # x1 = self.conv1(x1)
        # x2 = self.conv2(x2)
        x2 = F.interpolate(x2, size=x1.shape[-1], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x1.shape[-1], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x1.shape[-1], mode='bilinear', align_corners=True)
        # x2 = self.conv2(x2)
        # x3 = self.conv3(x3)
        # print('x1', x1.shape)
        # print('x2', x2.shape)
        # print('x3', x3.shape)
        # print('x4', x4.shape)
        x = torch.cat([x1, x2, x3, x4], 1)
        x = self.conv_(x)
        # x = self.conv_2(self.conv_1(x))
        # x = self.conv_1(x)
        x = self.conv_3(self.conv_2(self.conv_1(x)))
        return x

class StudentNet(nn.Module):
    def __init__(self, ed=True):
        super().__init__()
        self.ed = ed
        if self.ed:
            self.decoder_layer4 = make_layer(BasicBlock, 512, 512, 2)
            self.decoder_layer3 = make_layer(BasicBlock, 512, 256, 2)
            self.decoder_layer2 = make_layer(BasicBlock, 256, 128, 2)
            self.decoder_layer1 = make_layer(BasicBlock, 128, 64, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.encoder = timm.create_model(
            "resnet18",
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )
        self.multi_feature_fusion = MultiFeatureFusion()
    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        if not self.ed:
            return (x1, x2, x3)
        # x = x4
        x = self.multi_feature_fusion(x1, x2, x3, x4)
        b4 = self.decoder_layer4(x)
        b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
        b3 = self.decoder_layer3(b3)
        b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
        b2 = self.decoder_layer2(b2)
        b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
        b1 = self.decoder_layer1(b1)
        return (b1, b2, b3)

class ReNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1, self.bn1 = resnet18(pretrained=True)
        for name, param in self.encoder1.named_parameters():
            param.requires_grad = False
        self.decoder1 = de_resnet18(pretrained=False)

    def forward(self, x):
        self.encoder1.eval()
        inputs = self.encoder1(x)
        bn_outputs = self.bn1(inputs)
        outputs = self.decoder1(bn_outputs)
        return outputs


class fusionblock(nn.Module):
    def __init__(self, channels=128):
        super(fusionblock, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(channels // 2)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature0, feature1, feature2):
        feature0 = self.relu(self.bn0(self.conv0(feature0)))
        feature1 = self.relu(self.bn1(self.conv1(feature1)))
        feature2 = self.relu(self.bn2(self.conv2(feature2)))
        return (feature0, feature1, feature2)


class feed_back_fusion1(nn.Module):
    def __init__(self, channels=512):
        super(feed_back_fusion1, self).__init__()
        # self.feed = wide_resnet50_2(pretrained=False)[0]
        self.feed = StudentNet()
        self.back = ReNet()
        self.fusionblock = fusionblock()

    def forward(self, img_feed, img_back):
        feature_feed = self.feed(img_feed)
        feature_back = self.back(img_back)
        # print('feed', feature_feed[0].shape, feature_feed[1].shape, feature_feed[2].shape)
        # print('back', feature_back[0].shape, feature_back[1].shape, feature_back[2].shape)
        feature0 = torch.cat([feature_feed[0], feature_back[0]], 1)
        feature1 = torch.cat([feature_feed[1], feature_back[1]], 1)
        feature2 = torch.cat([feature_feed[2], feature_back[2]], 1)
        return self.fusionblock(feature0, feature1, feature2)

class SegmentationNet(nn.Module):
    def __init__(self, inplanes=448):
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x


class S2TKD(nn.Module):
    def __init__(self, dest=True, ed=True):
        super().__init__()
        self.teacher_net = TeacherNet()
        self.student_net = feed_back_fusion1()
        self.dest = dest
        self.segmentation_net = SegmentationNet(inplanes=448)

    def forward(self, img_aug, img_origin=None):
        self.teacher_net.eval()

        if img_origin is None:  # for inference
            img_origin = img_aug.clone()

        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_aug)
        ]
        outputs_student_aug = [
            # l2_normalize(output_s) for output_s in self.student_net(img_aug)
            l2_normalize(output_s) for output_s in self.student_net(img_aug, img_origin)
        ]
        # print('inputs:', outputs_teacher_aug[0].shape, outputs_teacher_aug[1].shape, outputs_teacher_aug[2].shape, 'outputs:', outputs_student_aug[0].shape,
        #       outputs_student_aug[1].shape, outputs_student_aug[2].shape)
        output = torch.cat(
            [
                F.interpolate(
                    -output_t * output_s,
                    size=outputs_student_aug[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
            ],
            dim=1,
        )
        # print('output', output.shape)
        output_segmentation = self.segmentation_net(output)

        if self.dest:
            outputs_student = outputs_student_aug
        else:
            outputs_student = [
                l2_normalize(output_s) for output_s in self.student_net(img_origin, img_origin)
            ]
        outputs_teacher = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_origin)
        ]

        output_de_st_list = []
        for output_t, output_s in zip(outputs_teacher, outputs_student):
            a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
            output_de_st_list.append(a_map)
        output_de_st = torch.cat(
            [
                F.interpolate(
                    output_de_st_instance,
                    size=outputs_student[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_de_st_instance in output_de_st_list
            ],
            dim=1,
        )  # [N, 3, H, W]
        output_de_st = torch.prod(output_de_st, dim=1, keepdim=True)

        return output_segmentation, output_de_st, output_de_st_list
