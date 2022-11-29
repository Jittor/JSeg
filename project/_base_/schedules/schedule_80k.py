max_iter = 80000
eval_interval = 4000
checkpoint_interval = 4000
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# scheduler
scheduler = dict(type='PolyLR',
                 max_steps=max_iter,
                 power=0.9,
                 min_lr=1e-4)
