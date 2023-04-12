from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
import os
import pickle as pkl

#vizualization function
def show_tensor_images(image_tensor, num_images=25, name=""):
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5).permute(1, 2, 0)
    plt.imshow(image_grid)
    #plt.grid(None)
    plt.show()

#gen noise
def get_noise(n_samples, zdim, device="cpu"):
    return torch.randn(n_samples, zdim, device=device)

#gradient penalty
def gradient_penalty(real, fake, crit, alpha, gama=10):
    #print(real.shape, fake.shape)
    mix_images = real * alpha + fake * (1 - alpha)
    mix_scores = crit(mix_images)
    gradient = torch.autograd.grad(
        inputs=mix_images,
        outputs=mix_scores,
        grad_outputs=torch.ones_like(mix_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = gama * ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty


def save_checkpoint(name, gen_model, gen_opt, crit_model, crit_opt, epoch):
    # save and load checkpoint json
    root_path = "data/checkpoints/"
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    print("=> Saving checkpoint")
    torch.save({
        "epoch": epoch,
        "gen_state_dict": gen_model.state_dict(),
        "gen_opt_state_dict": gen_opt.state_dict(),
    }, f"{root_path}G-{name}.pkl")
    torch.save({
        "epoch": epoch,
        "crit_state_dict": crit_model.state_dict(),
        "crit_opt_state_dict": crit_opt.state_dict(),
    }, f"{root_path}C-{name}.pkl")


def load_checkpoint(name, gen_model, gen_opt, crit_model, crit_opt):
    root_path = "data/checkpoints/"
    print("=> Loading checkpoint")
    checkpoint = torch.load(f"{root_path}G-{name}.pkl")
    gen_model.load_state_dict(checkpoint["gen_state_dict"])
    gen_opt.load_state_dict(checkpoint["gen_opt_state_dict"])

    checkpoint = torch.load(f"{root_path}C-{name}.pkl")
    crit_model.load_state_dict(checkpoint["crit_state_dict"])
    crit_opt.load_state_dict(checkpoint["crit_opt_state_dict"])
    print("=> Loaded checkpoint")
    return (gen_model, gen_opt, crit_model, crit_opt)