custom_hooks = [
    dict(begin_epoch=0, momentum=0.999, type='EMAHook'),
]
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        rule='greater',
        save_best='accuracy/top1',
        type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmselfsup'
env_cfg = dict(cudnn_benchmark=False)
log_processor = dict(
    by_epoch=True, type='mmengine.LogProcessor', window_size=1)
model = dict(
    backbone=dict(depth=18, in_channels=3, out_indices=(4, ), type='ResNet'),
    classifier=dict(
        act_cfg=dict(type='ReLU'),
        dropout_rate=0.2,
        in_channels=128,
        loss=dict(
            label_smooth_val=0.01,
            loss_weight=1,
            mode='original',
            num_classes=4,
            reduction='mean',
            type='mmcls.LabelSmoothLoss'),
        mid_channels=[
            64,
        ],
        num_classes=4,
        type='mmcls.StackedLinearClsHead'),
    head=dict(
        loss=dict(type='mmcls.CrossEntropyLoss'),
        temperature=0.1,
        type='ContrastiveHead'),
    neck=dict(
        hid_channels=64,
        in_channels=128,
        num_layers=2,
        out_channels=64,
        type='NonLinearNeck',
        with_avg_pool=False,
        with_last_bn=True),
    reducer=dict(
        in_channels=512,
        out_channels=128,
        type='SimpleReducer',
        with_avg_pool=True),
    regressor=dict(
        in_channels=128,
        loss=dict(loss_weight=1, reduction='mean', type='EuclideanLoss'),
        num_classes=1,
        type='mmcls.LinearClsHead'),
    type='SimCLRPlusClassifier')
optim_wrapper = dict(
    clip_grad=dict(max_norm=5.0, norm_type=2),
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(classifier=dict(lr_mult=1.0))),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.01, type='LinearLR'),
    dict(
        T_max=20,
        begin=5,
        by_epoch=True,
        end=20,
        eta_min=1e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=42)
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=256,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        h5_file=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/FISH/MYCN.h5',
        mode='regression',
        pipeline=[
            dict(
                num_views=[
                    2,
                ],
                transforms=[
                    [
                        dict(
                            p_horizontal=0.5,
                            p_vertical=0.5,
                            type='C_RandomFlip'),
                        dict(
                            angle=(
                                0,
                                360,
                            ),
                            order=1,
                            scale=(
                                0.6666666666666666,
                                1.5,
                            ),
                            shift=(
                                -0.1,
                                0.1,
                            ),
                            type='C_RandomAffine'),
                        dict(
                            high=(
                                3,
                                3,
                                3,
                            ),
                            low=(
                                0.5,
                                0.5,
                                0.5,
                            ),
                            type='C_RandomIntensity'),
                        dict(
                            high=(
                                0.3,
                                0.6,
                            ),
                            low=(
                                0,
                                0.3,
                            ),
                            threshold=0.1,
                            type='C_RandomGradient'),
                        dict(
                            blurr=(
                                0.0,
                                1,
                            ), clip=True, type='C_RandomBlurr'),
                        dict(
                            clip=True,
                            mean=(
                                -0.15,
                                0.15,
                            ),
                            std=(
                                0.0,
                                0.12,
                            ),
                            type='C_RandomNoise'),
                    ],
                ],
                type='MultiView'),
            dict(
                backend='numpy',
                dtype='float32',
                keys=[
                    'img',
                ],
                normalize_uint8_to_01=True,
                type='C_TypeCaster'),
            dict(
                pseudo_label_keys=[
                    'gt_label_class',
                    'gt_label_spots',
                ],
                type='PackSelfSupInputs'),
        ],
        type='PatchDataset'),
    drop_last=True,
    num_workers=32,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=128,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        h5_file=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/FISH/real_world_MYCN.h5',
        mode='classification',
        pipeline=[
            dict(
                backend='numpy',
                dtype='float32',
                keys=[
                    'img',
                ],
                normalize_uint8_to_01=True,
                type='C_TypeCaster'),
            dict(
                pseudo_label_keys=[
                    'gt_label_class',
                ],
                type='PackSelfSupInputs'),
        ],
        type='PatchDataset'),
    drop_last=True,
    num_workers=32,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(topk=(1, ), type='mmcls.Accuracy')
work_dir = './work_dirs/4class_MYCN'
