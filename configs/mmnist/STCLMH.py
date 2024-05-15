method = 'STCLMH'
# reverse scheduled sampling
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000
# scheduled sampling
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002
# model 

num_hidden = '256,256,256'
filter_size = 5
stride = 1
patch_size = 2
layer_norm = 1
# training
min_lr = 1e-4
batch_size = 16
sched = 'cosine'
warmup_epoch = 10
gradient_detach=0
# sched = 'onecycle'
