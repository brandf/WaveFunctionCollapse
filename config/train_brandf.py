# train a miniature character-level shakespeare model
# good or debugging and playing on macbooks and such

name = "noisy_prob_drop_0.01_196_0.75_0.5"

out_dir = 'out/' + name
eval_interval = 500
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'wfc'
wandb_run_name = name

dataset = 'openwebtext'

batch_size = 32
gradient_accumulation_steps = 2 # used to simulate larger batch sizes

block_size = 196 # context length

# baby GPT model :)
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.01

learning_rate = 1e-3
max_iters = 100000
lr_decay_iters = max_iters
min_lr = learning_rate / 5.0
beta2 = 0.99

warmup_iters = 100 # not super necessary potentially

# weight decay
weight_decay = 1e-1
