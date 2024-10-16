# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-stock-intraday-spaced'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 10
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'stocks'
wandb_run_name = 'gpt2'

dataset = 'C:\\Users\\catop\\Documents\\GitHub\\nanoGPT\\data\\stock_dat\\clean'
gradient_accumulation_steps = 2
batch_size = 32
block_size = 1024 

n_layer = 4
n_head = 4
n_embd = 384
dropout = 0.2

learning_rate = 5e-3 # with baby networks can afford to go a bit higher
max_iters = 1000
lr_decay_iters = 1000 # make equal to max_iters usually
min_lr = 5e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 10  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model