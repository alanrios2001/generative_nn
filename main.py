import torch, os
from models import Generator, Critic
from loader import load_dataset
from trainer import train
from functool import load_checkpoint

#hyperparameters and general parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-6
BATCH_SIZE = 64
zdim = 128
display_step = 1000

cur_step = 0
crit_cycles = 5
gen_losses = []
crit_losses = []
show_step = 35
save_step = 35
data_path = "data/imgnet"

#load dataset
dataloader = load_dataset(path=data_path, BATCH_SIZE=BATCH_SIZE)
print("Dataset loaded")
print("Number of batches: ", len(dataloader))
print("Number of images: ", len(dataloader)*BATCH_SIZE)
print("-"*25)


gen = Generator(zdim=zdim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
crit = Critic().to(device)
crit_opt = torch.optim.Adam(crit.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

gen, gen_opt, crit, crit_opt = load_checkpoint("model1",gen, gen_opt, crit, crit_opt)

print("Starting Training Loop...")
train(epochs=200,
      dataloader=dataloader,
      gen_model=gen,
      gen_opt=gen_opt,
      crit_model=crit,
      crit_opt=crit_opt,
      zdim=zdim,
      display_step=display_step,
      crit_cycles=crit_cycles,
      gen_losses=gen_losses,
      crit_losses=crit_losses,
      cur_step=cur_step,
      BATCH_SIZE=BATCH_SIZE,
      device=device)