import torch
from functool import get_noise, show_tensor_images, load_checkpoint
from models import Generator, Critic

device = "cuda" if torch.cuda.is_available() else "cpu"


zdim = 128
gen = Generator(zdim=zdim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=0, betas=(0.5, 0.999))
crit = Critic().to(device)
crit_opt = torch.optim.Adam(crit.parameters(), lr=0, betas=(0.5, 0.999))
gen, gen_opt, crit, crit_opt = load_checkpoint("model1",gen, gen_opt, crit, crit_opt)

noise = get_noise(25, zdim, device=device)
fake = gen(noise)
show_tensor_images(fake)