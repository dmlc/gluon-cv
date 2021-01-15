"""Model store which provides pretrained models."""
from __future__ import print_function

__all__ = ['get_model_file', 'purge']
import os
import zipfile
import logging
import portalocker

from ..utils import download, check_sha1

_model_sha1 = {name: checksum for checksum, name in [
    ('1a6f936097ef843c35b70ffd11e787bef94c530f', 'alexnet'),
    ('bb05f32ab708154b55bb74e80455ca0efb6aa8c4', 'densenet121'),
    ('d41edd9e5fee4fe933e23e84e3c4f620ee450121', 'densenet161'),
    ('7dc81e61e478be319ace2a3496fb2405f03c25aa', 'densenet169'),
    ('c375d600c13cb62dcf2b233714b6d670da100026', 'densenet201'),
    ('4461205f7595f580df06f5c785a383341a2e56f4', 'inceptionv3'),
    ('37c1c90b56800303a66934487fbf017bca8bba00', 'xception'),
    ('68c193f6a89b2d474b9bbb919f5b84f9b4ea9de9', 'mobilenet0.25'),
    ('cf5eb260b74dea9f456300c091b3e20832f1dbee', 'mobilenet0.5'),
    ('b415c1a0117cc02e79c6e2db62a19da3768e7f42', 'mobilenet0.75'),
    ('c6524b23cc839e727143ad8705d59275c62d64a6', 'mobilenet1.0'),
    ('ac895a1f9040c766563bec87ed36c41031cf6e39', 'mobilenetv2_1.0'),
    ('d74091d546be1bba385c6df528a92232c872de10', 'mobilenetv2_0.75'),
    ('a276bf4979109146c8b2aa2bde0de49c6b9810d3', 'mobilenetv2_0.5'),
    ('2b7d87801fa9a47de26ff700e0d294de6b04e7cd', 'mobilenetv2_0.25'),
    ('eaa44578554ddffaf2a2630ced9093181ff79688', 'mobilenetv3_large'),
    ('10430433698d18f49991e4a366c9fce8f9286298', 'mobilenetv3_small'),
    ('8a6d57bdb5aec46c34a499adbc7623930802ab10', 'resnet18_v1'),
    ('20ee8837b9151e88c13e952f6cfaf9d88ce11127', 'resnet34_v1'),
    ('3c1ce0981d5a4360c9c80706a0aeb1c9d93be7b7', 'resnet50_v1'),
    ('5a4b72986bb127ae700979399bf5e392207720d4', 'resnet101_v1'),
    ('82b3ec780b4c4bc93f7340172218c11c3997ebb0', 'resnet152_v1'),
    ('09e378c1fbb1f3cb87a186114d1f8e9b2534ca53', 'resnet18_v2'),
    ('93fbfb551e0ff321d4b6a9856f9a78bcfca3d096', 'resnet34_v2'),
    ('1b8139424aac7553bf5eed3baeeadecd9cc1e0fb', 'resnet50_v2'),
    ('07055b6544d3b7d395a30a1752362ae295b1ba3d', 'resnet101_v2'),
    ('7ea34e8a5cbe5f84047f120a81d737873443d03b', 'resnet152_v2'),
    ('d7c4a8e5ced4a0c9f112bec84ff947eca3839c9a', 'squeezenet1.0'),
    ('b84c551b655d2ad0b24515e255ba4eae8c04b8cd', 'squeezenet1.1'),
    ('f1d71970c974283ba56230f2de3a9fdebda775dc', 'vgg11'),
    ('2f8c77bec3c088d863cef9f43fd5e33138e478d6', 'vgg11_bn'),
    ('69e3b764ec46696c1cf46465e63a2e31773c8d86', 'vgg13'),
    ('8f53f594c939b55a7cb70d550af28d3db38954dd', 'vgg13_bn'),
    ('07007d12d8603e9e910db9514013f52629e568c0', 'vgg16'),
    ('cae35d3ea607857c1a0932a0ea95371db04b115e', 'vgg16_bn'),
    ('aa290c35c6b2a28a61c0056eb7727efb572ec995', 'vgg19'),
    ('cb708590ca25dd18a8de674239bdf51d7e538394', 'vgg19_bn'),
    ('4fa2e1ad96b8c8d1ba9e5a43556cd909d70b3985', 'vgg16_atrous'),
    ('397eb1ed7c66606377a321ef8c2546d847bde2dc', 'ssd_300_vgg16_atrous_voc'),
    ('d5215a6672263a75eec0ee746f6bcbb68dc315ae', 'ssd_300_resnet34_v1b_coco'),
    ('b87c7268e086367bba7694325a365db81c28ca1d', 'ssd_512_vgg16_atrous_voc'),
    ('000e02f3d4c31a59e1ff6b463e94a7497a9e9516', 'ssd_512_resnet50_v1_voc'),
    ('f991fdb96ee38f17fd6062f95d9f04642a6e636b', 'ssd_512_resnet101_v2_voc'),
    ('96772a96eaa124174a3ec471d4ee0175fe8977bf', 'ssd_512_mobilenet1.0_voc'),
    ('386e5e2aacdc2535fe217bda3a657e3e070b844f', 'ssd_300_vgg16_atrous_coco'),
    ('1ddae2d8854e64de5b1c07406f0fba9a75ec773d', 'ssd_512_vgg16_atrous_coco'),
    ('368a6be1378de56d20375ac2f3a7945ce9e1bb98', 'ssd_512_resnet50_v1_coco'),
    ('8e9dab1b12df9dfc0d72e5de7ac3c2f14d98f5e8', 'ssd_512_mobilenet1.0_coco'),
    ('5bfa313f394b8d7e154f9f2b09648e0a408ceb3b', 'faster_rcnn_resnet50_v1b_voc'),
    ('7b47b12ad23a4448351052e97cd503a9d458933b', 'faster_rcnn_resnet50_v1b_coco'),
    ('421c39412bcf3ffa6a6a04ae041423bc89e16be3', 'faster_rcnn_fpn_syncbn_resnest50_coco'),
    ('f8dc657a450839d59c81ccd2c04f1544e6e53427', 'faster_rcnn_resnet101_v1d_coco'),
    ('233572743bc537291590f4edf8a0c17c14b234bb', 'faster_rcnn_fpn_resnet50_v1b_coco'),
    ('1194ab4ec6e06386aadd55820add312c8ef59c74', 'faster_rcnn_fpn_resnet101_v1d_coco'),
    ('c77e3ae6573b727bc4f0f4c7f40a4e848946fd77', 'faster_rcnn_fpn_syncbn_resnest101_coco'),
    ('612906f52c4cb7f4864cab1b92edb10562df0c00', 'faster_rcnn_fpn_syncbn_resnest269_coco'),
    ('e071cf1550bc0331c218a9072b59e9550595d1e7', 'mask_rcnn_resnet18_v1b_coco'),
    ('c20a0e1c870e072abe401867401f5247b9985a47', 'mask_rcnn_resnet50_v1b_coco'),
    ('4a3249c584f81c2a9b5d852b742637cd692ebdcb', 'mask_rcnn_resnet101_v1d_coco'),
    ('14a922c38b0b196fdb1f7141be4666c10476f426', 'mask_rcnn_fpn_resnet18_v1b_coco'),
    ('1364d0afe4de575af5d4389d50c2dbf22449ceac', 'mask_rcnn_fpn_resnet50_v1b_coco'),
    ('89c7d8669b677a05c6eaa25375ead9a174109c69', 'mask_rcnn_fpn_resnet101_v1d_coco'),
    ('120451c565cdc4255c36d61037125bf423bea0f3', 'cifar_resnet20_v1'),
    ('827aad34003a0863e22befbccd40fc71ded9702a', 'cifar_resnet20_v2'),
    ('e3975f997287a2c8a619b51bdc6d1ca22c8ca83c', 'cifar_resnet56_v1'),
    ('5d9b00bfe6ad74b476fa9c3310d9d7124af9da56', 'cifar_resnet56_v2'),
    ('92a8868e9a0833ae42ab540e6e406527fae26fed', 'cifar_resnet110_v1'),
    ('1761633229b51425e60a41127d2dc76933f719c6', 'cifar_resnet110_v2'),
    ('da8e121f24d738a5b5e1315cef294e19ce97635c', 'cifar_wideresnet16_10'),
    ('cd645b79b5408283d126ee49eb9e981dcee6dd1c', 'cifar_wideresnet28_10'),
    ('6a0c9a17bf1d5028a0d1ff41e39927ea4c34803d', 'cifar_wideresnet40_8'),
    ('7394a7c822b37281b6f8af96c6a09203260d337b', 'cifar_resnext29_16x64d'),
    ('7e0b0cae2fd7e9084d1e498b858af1429a6eb1cb', 'resnest14'),
    ('364590740605b6a2b95f5bb77436d781a817436f', 'resnest26'),
    ('bcfefe1dd1dd1ef5cfed5563123c1490ea37b42e', 'resnest50'),
    ('5da943b3230f071525a98639945a6b3b3a45ac95', 'resnest101'),
    ('0c5d117df664ace220aa6fc2922c094bb079d381', 'resnest200'),
    ('11ae7f5da2bcdbad05ba7e84f9b74383e717f3e3', 'resnest269'),
    ('f3837de008e31e460a7a6f5ce315923c8a159c27', 'resnet18_v1b'),
    ('9e5e00845853640a844a373b55f46bfaf3e2bbe6', 'resnet34_v1b'),
    ('dd589a3d628b293a7bf9a9b4de1ed0a74c64e1ca', 'resnet50_v1b'),
    ('48ddf358d5acc879f76740dae695be67d96beea6', 'resnet50_v1b_gn'),
    ('db359f33ec24a3b870b546ea4d99d8f3e8cec158', 'resnet101_v1b'),
    ('093849afa589a05f3ff1d846a86d36a779bbd1eb', 'resnet152_v1b'),
    ('c338401bf71a958c1059b977faaee9936e428378', 'resnet50_v1c'),
    ('8a93f5591a486b714c9fe4dc755507f0517e0c98', 'resnet101_v1c'),
    ('900e886d47721c116aa5335adc2f4579d06497ed', 'resnet152_v1c'),
    ('2c309029709dc35f3a91883ebba10dbf7c8169b8', 'resnet50_v1d'),
    ('671f62851168d5b877e6ae9a8ea236d702d7e558', 'resnet101_v1d'),
    ('d6ea03df05a5c817ad4e945ab860dafff23dea5e', 'resnet152_v1d'),
    ('25a187fa281ddc98afbcd0cc0f0646885b874b80', 'resnet50_v1s'),
    ('c7a115c7096a16f2b653f94d349b685d99a089e6', 'resnet101_v1s'),
    ('9acd73cd4be32fd65c828e0afb429afb05405b98', 'resnet152_v1s'),
    ('766cdf9cc3e5b980b141643f054db5b48863f634', 'fcn_resnet101_coco'),
    ('12c2b9b3be7d4e133e52477150a9b3e616626a82', 'fcn_resnet101_voc'),
    ('3479525af7bdbf345e74e150aaae2e48174c0c5f', 'fcn_resnet50_ade'),
    ('d544440a35586f662ed1a5405ab9aa89cd750558', 'fcn_resnet101_ade'),
    ('ed817f76086abb4c3404af62ec1b5487c67642b7', 'deeplab_resnet101_coco'),
    ('311ed22c63f3ac28b5f1e1663c458f26600e62da', 'deeplab_resnet101_voc'),
    ('83247aaeeded6988790cd93ac93151514094846f', 'deeplab_resnet152_coco'),
    ('d35bea8817935d1ab310ef1e6dd06bb18c2d5f0d', 'deeplab_resnet152_voc'),
    ('c7789b237adc7253405bee57c84d53b15db45942', 'deeplab_resnet50_ade'),
    ('bf1584dfcec12063eff3075ee643e181c0f6d443', 'deeplab_resnet101_ade'),
    ('a8312db6e30a464151580f2bda83479786455724', 'deeplab_resnest50_ade'),
    ('6d05c630fb7acb38615f7f4d360fb90f47b25042', 'deeplab_resnest101_ade'),
    ('d45b33dedf4cca13b8147213c5360e30f93685bd', 'deeplab_resnest269_ade'),
    ('09f89cad0e107cb2bffdb1b07706ba31798096f2', 'psp_resnet101_coco'),
    ('2c2f4e1c2b11461b52598a4b2038bccbcfc166eb', 'psp_resnet101_voc'),
    ('3f220f537400dfa607c3d041ed3b172db39b0b01', 'psp_resnet50_ade'),
    ('240a4758b506447faf7c55cd7a7837d66f5039a6', 'psp_resnet101_ade'),
    ('d0e8603120ab02118a8973d52a26b8296d1b4078', 'psp_resnet101_citys'),
    ('ef2bb40ad8f8f59f451969b2fabe4e548394e80a', 'deeplab_v3b_plus_wideresnet_citys'),
    ('909742b45d5a3844d6000248aa92fef0ae23a0f0', 'icnet_resnet50_citys'),
    ('873d381a4bc246c5b9d3660ccf66c2f63d0b4e7c', 'icnet_resnet50_mhpv1'),
    ('cf6a7bb3d55360933de647a8505f7936003902a4', 'deeplab_resnet50_citys'),
    ('eb8477a91efc244c85b364c0736664078aaf0e65', 'deeplab_resnet101_citys'),
    ('95aad0b699ae17c67caa44b3ead4b23474e98954', 'fastscnn_citys'),
    ('143e1f1c1c1f2d3a3416887e088ebfdd4e1e2345', 'danet_resnet50_citys'),
    ('6ead3d099f7a320846bddb51148c3fe3b5ade5c2', 'danet_resnet101_citys'),
    ('75869793023128b12d5c394696b6bf9b8ebe4380', 'yolo3_darknet53_voc'),
    ('39879a23561959c150b56285534dc99a0fbdddf6', 'yolo3_mobilenet1.0_voc'),
    ('b4753994e85f89e4122bdb7e9a0cd0d63dc0c031', 'yolo3_mobilenet1.0_coco'),
    ('0f249e69c870d99e0a4bfe93e931ec58ff944e6e', 'yolo3_darknet53_coco'),
    ('304a931b086aea241c5b27a4fe7e9fc80ebfb420', 'darknet53'),
    ('7f7bfc509ce66e0f4e2ba0c747aa2fef7a8b305c', 'senet_154'),
    ('4ecf62e29336e0cbc5a2f844652635a330928b5a', 'resnext50_32x4d'),
    ('8654ca5d0ba30a7868c5b42a7d4cc0ff2ba04dbc', 'resnext101_32x4d'),
    ('2f0d1c9d343d140775bfa7548dd3a881a35855de', 'resnext101_64x4d'),
    ('7906e0e16013ef8d195cbc05463cc37783ec7a8a', 'se_resnext50_32x4d'),
    ('688e238985d45a38803c62cf345af2813d0e8aa0', 'se_resnext101_32x4d'),
    ('11c50114a0483e27e74dc4236904254ef05b634b', 'se_resnext101_64x4d'),
    ('c31bc54cf34414bb827b6f1c1c4584ba4173ca95', 'simple_pose_resnet18_v1b'),
    ('9ab77022d844b60acf0be8f8be8dcccbf09701b0', 'simple_pose_resnet50_v1b'),
    ('e14abd23b96acde412703b087b3487ef5f6fdcbf', 'simple_pose_resnet101_v1b'),
    ('264928f83c55722464b0f0ff87f8ca6d4ea83351', 'simple_pose_resnet152_v1b'),
    ('4387deabd6dca043ab0d1fc899f281008e081a8d', 'simple_pose_resnet50_v1d'),
    ('c705e6c99822b83ad72a8755e81764da3538638b', 'simple_pose_resnet101_v1d'),
    ('25b0dbe56cfb10281c6f57cfb995d4605d82d320', 'simple_pose_resnet152_v1d'),
    ('6dfcdcd1b48443c2067ec4143c061e40ed745914', 'mobile_pose_resnet18_v1b'),
    ('40d8b076cec42c8323eac4b19871aa98def3fde0', 'mobile_pose_resnet50_v1b'),
    ('6d56a2aa5beca7784624639e955f52d8ca6a76b5', 'mobile_pose_mobilenet1.0'),
    ('edb3a094e24ee7fe4b0d552262d4c737b599a2fb', 'mobile_pose_mobilenetv2_1.0'),
    ('0afe93b1e814280a5bb6014816d31bcb063981c6', 'mobile_pose_mobilenetv3_large'),
    ('c2a11fae8970c2c2e79e2b77a4c4d62e3d3e054f', 'mobile_pose_mobilenetv3_small'),
    ('54f7742b1f8939ef8e59ede3469bfa5eb6e247fa', 'resnet18_v1b_2.6x'),
    ('a230c33f7966ab761597328686b28d0545e4ea30', 'resnet50_v1d_1.8x'),
    ('0d3e69bb033d1375c3734419bbc653c3a474ea53', 'resnet50_v1d_3.6x'),
    ('9982ae4985b14e1c0ab25342a9f08bc4773b3998', 'resnet50_v1d_5.9x'),
    ('6a25eeceb7d27bd9c05fa2bf250c55d3960ad4c7', 'resnet50_v1d_8.8x'),
    ('a872796b63fb883116831db3454711421a628154', 'resnet101_v1d_1.9x'),
    ('712fccb185921a596baebe9246ff6c994b88591b', 'resnet101_v1d_2.2x'),
    ('de56b871543847d586deeca488b5bfe1b77bb5c5', 'alpha_pose_resnet101_v1b_coco'),
    ('96cd5b2449d6d9b3f8b64e3b5093928c2f0a9020', 'googlenet'),
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
    ('a007b5faf2927c5c4afe78db40756273f33a84e6', 'c3d_kinetics400'),
    ('671ba81c2b287ce3b9aa2393e15ff1562b705c4d', 'p3d_resnet50_kinetics400'),
    ('b30e3a6389c3a12867fe41bb3838f407d47aed2b', 'p3d_resnet101_kinetics400'),
    ('5a14d1f9772e523bbeae8484ec382feb30b74ffa', 'r2plus1d_resnet18_kinetics400'),
    ('de2e592b450b9007705507d5167859d45e782695', 'r2plus1d_resnet34_kinetics400'),
    ('deaefb14d31c703115a911828534d24ac68c7fb0', 'r2plus1d_resnet50_kinetics400'),
    ('81e0be1043ea4c68a4a0a439918c7b888ee5545b', 'i3d_inceptionv1_kinetics400'),
    ('f14f8a99007200ef56521b6a844f662784a8f5de', 'i3d_inceptionv3_kinetics400'),
    ('760d0981094787b8789ee4a8c382d09d493c7413', 'i3d_resnet50_v1_ucf101'),
    ('2ec6bf01a55af38579380e6531d0ecc816862abe', 'i3d_resnet50_v1_hmdb51'),
    ('568a722eb61da663e11b582886ddbef9ef8f6ac6', 'i3d_resnet50_v1_kinetics400'),
    ('01961e4cccf6405cd1342670b9525c21c578c9d4', 'i3d_resnet50_v1_sthsthv2'),
    ('6b69f655c60823bd05a83fe076c61d6c297add0d', 'i3d_resnet101_v1_kinetics400'),
    ('3c0e47ea5ee699c3e1f706c9df7a74dbd6321b11', 'i3d_nl5_resnet50_v1_kinetics400'),
    ('bfb58c4127705ad6e98f4916abde0c849e2f1288', 'i3d_nl10_resnet50_v1_kinetics400'),
    ('fbfc1d30d90c304295dedd6c70b037d100e43d5f', 'i3d_nl5_resnet101_v1_kinetics400'),
    ('59186c31dea2f20940a358fc8ea5199cd6d4303c', 'i3d_nl10_resnet101_v1_kinetics400'),
    ('9d650f5186ffc08348f8d7384d6994a8f39a03b1', 'slowfast_4x16_resnet50_kinetics400'),
    ('d6b253398615a21d8b2a827ddfb09c0d8827f79c', 'slowfast_8x8_resnet50_kinetics400'),
    ('fbde1a7cfdaeeba7190dd15f284b91036bf5f3f6', 'slowfast_8x8_resnet101_kinetics400'),
    ('5438e132948b5b2b6312aa8cb1dc1364769e436c', 'dla34'),
    ('096b5b9ad888dc9df7248e0f9ed9567544575ef8', 'center_net_resnet18_v1b_voc'),
    ('5062eda692b38a5c836d105826c24bb2ded29d8f', 'center_net_resnet18_v1b_dcnv2_voc'),
    ('fd56c60e136baf500c4d0815fea1589b1b347dba', 'center_net_resnet50_v1b_voc'),
    ('c48821eff51709b4df218c45d1897b5b97c2b159', 'center_net_resnet50_v1b_dcnv2_voc'),
    ('96621a18ee9a2b0085c71aa3e5248ae1705cd7cc', 'center_net_resnet101_v1b_voc'),
    ('575a46a5fead3b3bdf317312fb7566f5ea7d7992', 'center_net_resnet101_v1b_dcnv2_voc'),
    ('fa5b2badf0f4fa542c23e5a44789dfc06bab1ad5', 'center_net_resnet18_v1b_coco'),
    ('bbc452808d410f7dcced164fb034dd583e1a2f05', 'center_net_resnet18_v1b_dcnv2_coco'),
    ('31c31bad5611fb5d3d2d8c9997b4f7c8c9f933a2', 'center_net_resnet50_v1b_coco'),
    ('e505908e094512367f312af8345a280c4c376a10', 'center_net_resnet50_v1b_dcnv2_coco'),
    ('a1fb01c97567365651152d4fec443f124aeafdf8', 'center_net_resnet101_v1b_coco'),
    ('b4dd7dfba8d2e49c28f8f540c7400a1b4a6db586', 'center_net_resnet101_v1b_dcnv2_coco'),
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
    ('83eea4a9dda3323dca4b11c34c2d0c557056d7b5', 'monodepth2_resnet18_kitti_stereo_640x192'),
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
