"""Model store which provides pretrained models."""
from __future__ import print_function

__all__ = ['get_model_file', 'purge']
import os
import zipfile
import logging
import portalocker

from ..utils import download, check_sha1

_model_sha1 = {name: checksum for checksum, name in [
    ('401299ba982928855c0faf843867266afe4e0c98', 'alexnet'),
    ('4162624af65de0f77b780c0a45f494cb5c870ec6', 'densenet121'),
    ('973f8109a48b2fe0da957c4ab4c305b54073e526', 'densenet161'),
    ('18d12ea8399efb7c9b24d300965e0d8367a58d03', 'densenet169'),
    ('c9d2ee6318c0876e107d2163b629c9067aa7760d', 'densenet201'),
    ('6556d3f91b552e8fb239941368445f555b87d23e', 'inceptionv3'),
    ('3b96c7f9d6fafe99647e7c0d524d186ca5ec5858', 'xception'),
    ('b689d6fd8551c5204a22007482dd7818e968163f', 'mobilenet0.25'),
    ('1b93aee2da06224c1f1749cbe8a0c4efb76bb752', 'mobilenet0.5'),
    ('92ff4fb21ce5b3eaa3eaa03e971fd21625000fd4', 'mobilenet0.75'),
    ('adbf43036ab31c78ad5ac09b3ae3b8e220738f94', 'mobilenet1.0'),
    ('2421785d701b5b06f35da652f0484992ea497f34', 'mobilenetv2_1.0'),
    ('469c83cc6dfd91153363c09ad0a2ad7d5333bc87', 'mobilenetv2_0.75'),
    ('97be96ffaca98ec2a6658c9e911b54b97034ee30', 'mobilenetv2_0.5'),
    ('a3b830c5ccbf2a7449316e49dac2d2c51e7bf42c', 'mobilenetv2_0.25'),
    ('d3980b3594626340cef9218cf7097f25f93867e8', 'mobilenetv3_large'),
    ('b56c467558b49635ee3d661f86259839595ffbf3', 'mobilenetv3_small'),
    ('6b20be149e1d148f07766a55326666cd9a87ad50', 'resnet18_v1'),
    ('fe0a3f91447d6a0b8057d19388499cfa8486b1bd', 'resnet34_v1'),
    ('08dea4fa19a270711ed33dff27906b0d7572504d', 'resnet50_v1'),
    ('0f01e8b15552437cd2112752497c589800730d2d', 'resnet101_v1'),
    ('fb396ca05f22908f7cd4e193d79b4552ce536d07', 'resnet152_v1'),
    ('d5c337ed5d2e964d5430b88d0a31b6114a0b2685', 'resnet18_v2'),
    ('03e0a5f809308a5a1a34869a069566859e720e6e', 'resnet34_v2'),
    ('148846d137b7d191cb6adc57cfca64cee8956d23', 'resnet50_v2'),
    ('1992b5b1e2bec18d0ec940981297413724dc7b78', 'resnet101_v2'),
    ('6f1b7341070c1987980891ba54d7a6bcbc66b16c', 'resnet152_v2'),
    ('12081827f0415be573be8ba59f9edfcce463226d', 'squeezenet1.0'),
    ('7e0118053d83afd7094c3af1e38cb13f65ef7cfe', 'squeezenet1.1'),
    ('559be81bcd2b12137e0b2284e2b371241863bfe4', 'vgg11'),
    ('6a5b45d2a9a79a889b566ac2cb57cd116ce07357', 'vgg11_bn'),
    ('d3f5e7a96e0970f4aa9bd8974c66c40955c97348', 'vgg13'),
    ('1c1e6d7c37cdc2fd07d293b1838a1684b9e24051', 'vgg13_bn'),
    ('e3e9005752b043d0ef39fc457fa9fdb8316908f3', 'vgg16'),
    ('f947958fec63d5c7119a4339795addbab26cd4c0', 'vgg16_bn'),
    ('88bceba39bea049e3ce50b3aec9582ba39f92573', 'vgg19'),
    ('3cfe05aec5f9822807b1bde23886ff12ddd16feb', 'vgg19_bn'),
    ('4fa2e1ad96b8c8d1ba9e5a43556cd909d70b3985', 'vgg16_atrous'),
    ('1a4f837997e30da728ffb2cf80ced504b5435fad', 'ssd_300_vgg16_atrous_voc'),
    ('ade34ff72a2418fac94afe1048bc305b980c83c5', 'ssd_300_resnet34_v1b_coco'),
    ('496f80d9068f4a4711c4ad3ca684f5d6bbc2c1c3', 'ssd_512_vgg16_atrous_voc'),
    ('5756b248f4275149029a94188fce8e940ef200da', 'ssd_512_resnet50_v1_voc'),
    ('0218b47470c73282611c76dd54790e443806c36f', 'ssd_512_resnet101_v2_voc'),
    ('9324b04f0f21bc165ce3719c047834dc26403410', 'ssd_512_mobilenet1.0_voc'),
    ('e3b50cf9290c39c94748fc0fbd4bd6617492942c', 'ssd_300_vgg16_atrous_coco'),
    ('023891b37a89c6d909ff1ee651fa280f139798f7', 'ssd_512_vgg16_atrous_coco'),
    ('c30f830c0d1ce083e25c2317c38a5299065eae07', 'ssd_512_resnet50_v1_coco'),
    ('e3a85896883f22d31d52d3e64b5bd17fdd824d46', 'ssd_512_mobilenet1.0_coco'),
    ('447328d89d70ae1e2ca49226b8d834e5a5456df3', 'faster_rcnn_resnet50_v1b_voc'),
    ('5b4690fb7c5b62c44fb36c67d0642b633697f1bb', 'faster_rcnn_resnet50_v1b_coco'),
    ('6df46961827647d418b11ffaf616a6a60d9dd16e', 'faster_rcnn_fpn_syncbn_resnest50_coco'),
    ('a465eca35e78aba6ebdf99bf52031a447e501063', 'faster_rcnn_resnet101_v1d_coco'),
    ('233572743bc537291590f4edf8a0c17c14b234bb', 'faster_rcnn_fpn_resnet50_v1b_coco'),
    ('1194ab4ec6e06386aadd55820add312c8ef59c74', 'faster_rcnn_fpn_resnet101_v1d_coco'),
    ('baebfa1b7d7f56dd33a7687efea4b014736bd791', 'faster_rcnn_fpn_syncbn_resnest101_coco'),
    ('b7d778f58921e459a6af200f6a323d6fe67069b9', 'faster_rcnn_fpn_syncbn_resnest269_coco'),
    ('e071cf1550bc0331c218a9072b59e9550595d1e7', 'mask_rcnn_resnet18_v1b_coco'),
    ('a3527fdc2cee5b1f32a61e5fd7cda8fb673e86e5', 'mask_rcnn_resnet50_v1b_coco'),
    ('4a3249c584f81c2a9b5d852b742637cd692ebdcb', 'mask_rcnn_resnet101_v1d_coco'),
    ('14a922c38b0b196fdb1f7141be4666c10476f426', 'mask_rcnn_fpn_resnet18_v1b_coco'),
    ('1364d0afe4de575af5d4389d50c2dbf22449ceac', 'mask_rcnn_fpn_resnet50_v1b_coco'),
    ('89c7d8669b677a05c6eaa25375ead9a174109c69', 'mask_rcnn_fpn_resnet101_v1d_coco'),
    ('6280b54d079aa50b01277004210bccc0dd9cc790', 'cifar_resnet20_v1'),
    ('db3db5d7af7c6f543902198c0a98ce260fe5e6ad', 'cifar_resnet20_v2'),
    ('5443d707e350756cd57671e8735d2def570e7452', 'cifar_resnet56_v1'),
    ('470cdae9e0c731e19421138a0f82d40d1d8106f4', 'cifar_resnet56_v2'),
    ('881b96ffeed2cd3459b27ba2c4079b8254630017', 'cifar_resnet110_v1'),
    ('fad319b1111b471879a6422492b7799f82ebd5ed', 'cifar_resnet110_v2'),
    ('3529345222663838196f2d5e4976eff5e613b030', 'cifar_wideresnet16_10'),
    ('64fc2184f9d839b053e32a328a87c60c15440ffb', 'cifar_wideresnet28_10'),
    ('36e178a96ee794994c6040c81d4382c447949705', 'cifar_wideresnet40_8'),
    ('66b3bf6a7a0477a52f56bce643cbc1f0d9705653', 'cifar_resnext29_16x64d'),
    ('a914030d593eed699322ef9c6de8963e406fb75d', 'resnest14'),
    ('ef33105fb2676e484e84341b8be4ee6ff31f5234', 'resnest26'),
    ('2aac91dbffee2fe6acc6b9d43835269332be4b2b', 'resnest50'),
    ('72d66eefd8363bb16808a146effa1469f905e1e2', 'resnest101'),
    ('302f3d4a7d2276cb13020889edba70e910bb5e95', 'resnest200'),
    ('fbbf24954d24e4eacb32e3d3d7432dd0b3b09e1d', 'resnest269'),
    ('7aee6f040bde0eb06e32164e924c386d0ca15395', 'resnet18_v1b'),
    ('d0166227e5ec94a1888f27e01d2d65e7f4351a85', 'resnet34_v1b'),
    ('689d6119b8f4eaca507a8d9073d3036d84fe916a', 'resnet50_v1b'),
    ('dbc9c1d1874bab878b18e20be359f9c69da834ce', 'resnet50_v1b_gn'),
    ('0e45917b08ef518b69432b85b6a7be34023a2e41', 'resnet101_v1b'),
    ('ce9ef9bfac83177c6d6cbc47d8736721193a052b', 'resnet152_v1b'),
    ('276b2230cbfbfe7cd0dc75117235baf99f40f54d', 'resnet50_v1c'),
    ('5f6892f51a9cd9f8bfbd6ff1301e7a7ddb306560', 'resnet101_v1c'),
    ('689d0ea4d2460ee2f63da26f57de28e9364a615f', 'resnet152_v1c'),
    ('1c0abdc84431fc181dfda720db26edd2f16e10ac', 'resnet50_v1d'),
    ('40dc84382f25189f85b70cf0bfb61d7838ce47c6', 'resnet101_v1d'),
    ('70a5fab252dd06ac87a70c0bce5d1319bb6a983a', 'resnet152_v1d'),
    ('ea5cd5bcdac8980aa60e4c986e21aa7a51fbf9cd', 'resnet50_v1s'),
    ('259b7d85ae47abd22d2375a3ee96f9c1807f98f6', 'resnet101_v1s'),
    ('bb05e70c5e0172c26aa305fcbcb87fa34c57054d', 'resnet152_v1s'),
    ('0e4e93bc212d786ee26f3c5a68c4a2356c4db5be', 'fcn_resnet101_coco'),
    ('b08feff07a8e16da236e9e06ec02dba093040f80', 'fcn_resnet101_voc'),
    ('bd44319d0dd19125638a582d2456028530563927', 'fcn_resnet50_ade'),
    ('06a5d261f5928b19ab99d62470e6fa9d7f70b61a', 'fcn_resnet101_ade'),
    ('23f75d42862539d51c1df427b98473633840b358', 'deeplab_resnet101_coco'),
    ('ec5d20ca72ab4bc404e8b2c58424cb9f2307b66a', 'deeplab_resnet101_voc'),
    ('da8b7503409a062c78078dd2eaf32b20aa1cc366', 'deeplab_resnet152_coco'),
    ('dbbbab297a7626e4694b7dee6252c1ba2253563c', 'deeplab_resnet152_voc'),
    ('a438b9e119f6652865b4442353d23dd9d33de1ed', 'deeplab_resnet50_ade'),
    ('2350fb03836e7ccb595c5eeea7ff1faffc145602', 'deeplab_resnet101_ade'),
    ('4b0a4869170a12e7a21437650a5a2eb66eb250e7', 'deeplab_resnest50_ade'),
    ('c5c9b6744b13034661eeccbdb52343646e091b26', 'deeplab_resnest101_ade'),
    ('7f8628ff4c371ead7558d1d3c1dbe1d3382f4342', 'deeplab_resnest269_ade'),
    ('8f47084706c107a50a4931c5f5a0dd0f7f88014e', 'psp_resnet101_coco'),
    ('0fd5ad39d355623927c829fa764c06771aa7460d', 'psp_resnet101_voc'),
    ('0e32914062d32668f26853c096a5e631adf41b99', 'psp_resnet50_ade'),
    ('5c67c2d2e255f3409db048f5c84a9af1e0b1c5c2', 'psp_resnet101_ade'),
    ('22625f4542b828935fd1b3b3438db0e0d061e89c', 'psp_resnet101_citys'),
    ('a86654e801cf7e1ef911a1fb7376ab54dc470b8d', 'deeplab_v3b_plus_wideresnet_citys'),
    ('ac9e93983c923eb852e10de685c36373db25b8d5', 'icnet_resnet50_citys'),
    ('48ce65285fb2824245783a610656baece84b1b57', 'icnet_resnet50_mhpv1'),
    ('3da7e9816df0c8df1ee59ec7929e3a9d01dbf309', 'deeplab_resnet50_citys'),
    ('fba798a223e11ef0bce5f74a978b64dd5611f83f', 'deeplab_resnet101_citys'),
    ('18ca126c4b94a00437a402253e5cce88ab214967', 'fastscnn_citys'),
    ('143e1f1c1c1f2d3a3416887e088ebfdd4e1e2345', 'danet_resnet50_citys'),
    ('6ead3d099f7a320846bddb51148c3fe3b5ade5c2', 'danet_resnet101_citys'),
    ('f5ece5ce1422eeca3ce2908004e469ffdf91fd41', 'yolo3_darknet53_voc'),
    ('3b47835ac3dd80f29576633949aa58aee3094353', 'yolo3_mobilenet1.0_voc'),
    ('66dbbae67be8f1e3cd3c995ce626a2bdc89769c6', 'yolo3_mobilenet1.0_coco'),
    ('09767802230b45af1c27697a2dad6d1ebaacc1e2', 'yolo3_darknet53_coco'),
    ('d4e091548e963a140fc48c97f67f4e6aa953ae18', 'darknet53'),
    ('b050e87ea02e37780f97ad1deab26cf341919f4b', 'senet_154'),
    ('842820d21b8930af9a6b9616e0c45f027b870ab3', 'resnext50_32x4d'),
    ('e6ec6cc920b4bb238223371f8a9c9cbd98fc917e', 'resnext101_32x4d'),
    ('b56230e93597fd9fcf2e3695a076d469ac8b8174', 'resnext101_64x4d'),
    ('746f88a53a8fe465dc3f0a494d6e72135ea58494', 'se_resnext50_32x4d'),
    ('291b7aaf485a2bb899a87352c9fbb1744a1e873a', 'se_resnext101_32x4d'),
    ('be15142f0461acd5076eff0c9ec949ee7cdb4a14', 'se_resnext101_64x4d'),
    ('f63d42ac8f83b239d4e08b636b888b8e50cd066d', 'simple_pose_resnet18_v1b'),
    ('e2c7b1adea31264bc9220511308b4efa89c6fc50', 'simple_pose_resnet50_v1b'),
    ('b7ec0de1a34eb718efd4a84339cc1547ead88cbe', 'simple_pose_resnet101_v1b'),
    ('ef4e033612a5fca6fc69e54c87da3ba3866d533e', 'simple_pose_resnet152_v1b'),
    ('ba2675b6a43fc31601f0e99311b0bb115369bc82', 'simple_pose_resnet50_v1d'),
    ('1f8f48fd49a23bcc73c1cd736bdc639cd1434489', 'simple_pose_resnet101_v1d'),
    ('3ca502ea8eaaa15f4f972d5cf139167d15ffa798', 'simple_pose_resnet152_v1d'),
    ('a059bac1b0324b720c397be712e95e55043363c5', 'mobile_pose_resnet18_v1b'),
    ('338d9e58c93afd1de7dac7a0885ce4c452492e78', 'mobile_pose_resnet50_v1b'),
    ('7721a9a2515b4ac8fc4d184dff16a77586a7e646', 'mobile_pose_mobilenet1.0'),
    ('1efcbe727aa2724a928fc4059734df972b15aff4', 'mobile_pose_mobilenetv2_1.0'),
    ('83c1be888178c8b98b07cb4bca20f2b8fcad565d', 'mobile_pose_mobilenetv3_large'),
    ('3588ffc8cd2cdfd13ebe33617caa455cf93cd109', 'mobile_pose_mobilenetv3_small'),
    ('54f7742b1f8939ef8e59ede3469bfa5eb6e247fa', 'resnet18_v1b_2.6x'),
    ('a230c33f7966ab761597328686b28d0545e4ea30', 'resnet50_v1d_1.8x'),
    ('0d3e69bb033d1375c3734419bbc653c3a474ea53', 'resnet50_v1d_3.6x'),
    ('9982ae4985b14e1c0ab25342a9f08bc4773b3998', 'resnet50_v1d_5.9x'),
    ('6a25eeceb7d27bd9c05fa2bf250c55d3960ad4c7', 'resnet50_v1d_8.8x'),
    ('a872796b63fb883116831db3454711421a628154', 'resnet101_v1d_1.9x'),
    ('712fccb185921a596baebe9246ff6c994b88591b', 'resnet101_v1d_2.2x'),
    ('de56b871543847d586deeca488b5bfe1b77bb5c5', 'alpha_pose_resnet101_v1b_coco'),
    ('5b21e32a94da98c7dde36e96c36963ff808700aa', 'googlenet'),
    ('d6dc1bbaaf3cbe0be19f02362706393f62ce76fa', 'vgg16_ucf101'),
    ('6dcdafb1dd19866fdd3116cbe3689adb85824b2c', 'inceptionv1_kinetics400'),
    ('13ef5c3bd40141f11c5b4e471f8f2ad0c1ac6299', 'inceptionv3_ucf101'),
    ('8a4a6946893de23937b5e5e4a1a449800d57ff71', 'inceptionv3_kinetics400'),
    ('46d5a9850cdb884eb9c9a95d48269c3d60c94d77', 'resnet18_v1b_kinetics400'),
    ('8a8d0d8d395882f836d379aefb4136dce8763d0c', 'resnet34_v1b_kinetics400'),
    ('682591e23ce4b92fbd3222c0710ebb52166fddca', 'resnet50_v1b_hmdb51'),
    ('cc757e5c94fdaaea64a8cb5acfbd655ddf6ffa96', 'resnet50_v1b_kinetics400'),
    ('80ee0c6bef8b0240ec88273531cd7f43e3f6b65d', 'resnet50_v1b_sthsthv2'),
    ('5bb6098ea5343dc0f4bf8536b3771b8ddfe231d7', 'resnet101_v1b_kinetics400'),
    ('9bc70c66059854a22e935b42f783ef2bec0377d8', 'resnet152_v1b_kinetics400'),
    ('abe53331539ac2e1c8462ded63050f8c981b4993', 'c3d_kinetics400'),
    ('41c99279053d5e1f29b9926560e380d358639e54', 'p3d_resnet50_kinetics400'),
    ('ba1c16ed0618546bb53012f410c0989e60841ae5', 'p3d_resnet101_kinetics400'),
    ('694371325c8cd618efbeb22707d5e27896115b6d', 'r2plus1d_resnet18_kinetics400'),
    ('7db69edbbf79e350d9be81b92d38b5a51f5fe4e9', 'r2plus1d_resnet34_kinetics400'),
    ('a66fd9e99d5770a59b254749c484780d7b35858b', 'r2plus1d_resnet50_kinetics400'),
    ('b7fb17df37b95504d0fd1ce87a175b0000464cb5', 'i3d_inceptionv1_kinetics400'),
    ('2545c14640b00d9a96735d50d8d973484df371e2', 'i3d_inceptionv3_kinetics400'),
    ('760d0981094787b8789ee4a8c382d09d493c7413', 'i3d_resnet50_v1_ucf101'),
    ('2ec6bf01a55af38579380e6531d0ecc816862abe', 'i3d_resnet50_v1_hmdb51'),
    ('568a722eb61da663e11b582886ddbef9ef8f6ac6', 'i3d_resnet50_v1_kinetics400'),
    ('01961e4cccf6405cd1342670b9525c21c578c9d4', 'i3d_resnet50_v1_sthsthv2'),
    ('6b69f655c60823bd05a83fe076c61d6c297add0d', 'i3d_resnet101_v1_kinetics400'),
    ('3c0e47ea5ee699c3e1f706c9df7a74dbd6321b11', 'i3d_nl5_resnet50_v1_kinetics400'),
    ('bfb58c4127705ad6e98f4916abde0c849e2f1288', 'i3d_nl10_resnet50_v1_kinetics400'),
    ('fbfc1d30d90c304295dedd6c70b037d100e43d5f', 'i3d_nl5_resnet101_v1_kinetics400'),
    ('59186c31dea2f20940a358fc8ea5199cd6d4303c', 'i3d_nl10_resnet101_v1_kinetics400'),
    ('00529aebfdd3b4ea41abe0fd0695ae7ba9d5e536', 'slowfast_4x16_resnet50_kinetics400'),
    ('3a43c60ff476e1abe5a869dffa847285a94faab1', 'slowfast_8x8_resnet50_kinetics400'),
    ('dbcc4021168a0d84794999229649159285b45363', 'slowfast_8x8_resnet101_kinetics400'),
    ('5fd8d651eea4b8f3767174ea63bd0afc1fa541d8', 'dla34'),
    ('38c509d456a6e14e4c54e961ed43bffe8cf62840', 'center_net_resnet18_v1b_voc'),
    ('04d1ed20ae3f8a150bc06d8007775a6df99e40b2', 'center_net_resnet18_v1b_dcnv2_voc'),
    ('f108427bc62c85f7bcbdba45db4c94a31fd4d4f6', 'center_net_resnet50_v1b_voc'),
    ('61eb866d36ef68b1145a74a5b4e76ba262dbb4e4', 'center_net_resnet50_v1b_dcnv2_voc'),
    ('5bf8b91f8813e82a2f2660c83560ffdfebf835a0', 'center_net_resnet101_v1b_voc'),
    ('a0e707225164fc578b5bb7187a79e6b492da8fb4', 'center_net_resnet101_v1b_dcnv2_voc'),
    ('dccae71d1f069343326750bc9b0508a2a760dd80', 'center_net_resnet18_v1b_coco'),
    ('0874df9a5236297fd32ed401f0a699602ee6b42b', 'center_net_resnet18_v1b_dcnv2_coco'),
    ('28c64aaeaf9d5e4404afd2b96bf8812973d79eb9', 'center_net_resnet50_v1b_coco'),
    ('2713a7ba29ab4da5f1939da5a53f97ed079441ca', 'center_net_resnet50_v1b_dcnv2_coco'),
    ('6f9cd4a945cb554c38539009eede20f45786519e', 'center_net_resnet101_v1b_coco'),
    ('52daf2d9faca82c08924d08e7e0c253a782a1653', 'center_net_resnet101_v1b_dcnv2_coco'),
    ('df6ae9f896cabf5e47d39294d0d78282192d3c24', 'siamrpn_alexnet_v2_otb15'),
    ('9c946e56229ae6d1e4ec482c3be63ae2ee33d654', 'hrnet_w18_cls'),
    ('c224ae0fe80dd060d244b6a934dada1de0a9c021', 'hrnet_w18_small_v1_cls'),
    ('346da53060b949a3a3ffb83a9b6f9ba46c5c4fbc', 'hrnet_w18_small_v2_cls'),
    ('2d8eb90b4a3dc8a4fc8d95910e83de425f002c74', 'hrnet_w30_cls'),
    ('d1d41a36de3d1eca7b595973779268bc3f828f5b', 'hrnet_w32_cls'),
    ('f8b0a2f9e2db56d23c594847b7f913db9e56cc57', 'hrnet_w40_cls'),
    ('b36ab13a611bfd196c46dc9cdda918b63789722d', 'hrnet_w44_cls'),
    ('be50e56dea846e87343b09b4ba9687b98bd47d0c', 'hrnet_w48_cls'),
    ('76f1b7077179f21c4147ff77f9e0999db879d7ec', 'hrnet_w64_cls'),
    ('8a8e9cbcbac3a496f0904dc22e148b1978ea672f', 'center_net_mobilenetv3_small_duc_voc'),
    ('f782057569a768faa03940d8106f944ffe3c9733', 'center_net_mobilenetv3_small_duc_coco'),
    ('2cab979937586d3f8532d86130030c611ae38d2d', 'center_net_mobilenetv3_large_duc_voc'),
    ('bfc55dfd13ef8e9fd052141a3c42338179f5949d', 'center_net_mobilenetv3_large_duc_coco'),
    ('88435f66bcf74bd200b287ba6bc57f5b1c7260fd', 'monodepth2_resnet18_kitti_stereo_640x192'),
]}

apache_repo_url = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'
_url_format = '{repo_url}gluon/models/{file_name}.zip'


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]


def get_model_file(name, tag=None, root=os.path.join('~', '.mxnet', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    if 'MXNET_HOME' in os.environ:
        root = os.path.join(os.environ['MXNET_HOME'], 'models')

    use_tag = isinstance(tag, str)
    if use_tag:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=tag)
    else:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=short_hash(name))
    root = os.path.expanduser(root)
    params_path = os.path.join(root, file_name + '.params')
    lockfile = os.path.join(root, file_name + '.lock')
    if use_tag:
        sha1_hash = tag
    else:
        sha1_hash = _model_sha1[name]

    if not os.path.exists(root):
        os.makedirs(root)

    with portalocker.Lock(lockfile, timeout=int(os.environ.get('GLUON_MODEL_LOCK_TIMEOUT', 300))):
        if os.path.exists(params_path):
            if check_sha1(params_path, sha1_hash):
                return params_path
            else:
                logging.warning("Hash mismatch in the content of model file '%s' detected. "
                                "Downloading again.", params_path)
        else:
            logging.info('Model file not found. Downloading.')

        zip_file_path = os.path.join(root, file_name + '.zip')
        repo_url = os.environ.get('MXNET_GLUON_REPO', apache_repo_url)
        if repo_url[-1] != '/':
            repo_url = repo_url + '/'
        download(_url_format.format(repo_url=repo_url, file_name=file_name),
                 path=zip_file_path,
                 overwrite=True)
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall(root)
        os.remove(zip_file_path)
        # Make sure we write the model file on networked filesystems
        try:
            os.sync()
        except AttributeError:
            pass
        if check_sha1(params_path, sha1_hash):
            return params_path
        else:
            raise ValueError('Downloaded file has different hash. Please try again.')


def purge(root=os.path.join('~', '.mxnet', 'models')):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".params"):
            os.remove(os.path.join(root, f))


def pretrained_model_list():
    """Get list of model which has pretrained weights available."""
    _renames = {
        'resnet18_v1b_2.6x': 'resnet18_v1b_0.89',
        'resnet50_v1d_1.8x': 'resnet50_v1d_0.86',
        'resnet50_v1d_3.6x': 'resnet50_v1d_0.48',
        'resnet50_v1d_5.9x': 'resnet50_v1d_0.37',
        'resnet50_v1d_8.8x': 'resnet50_v1d_0.11',
        'resnet101_v1d_1.9x': 'resnet101_v1d_0.76',
        'resnet101_v1d_2.2x': 'resnet101_v1d_0.73',
    }
    return [_renames[x] if x in _renames else x for x in _model_sha1.keys()]
