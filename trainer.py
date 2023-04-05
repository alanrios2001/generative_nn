import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from functool import get_noise, gradient_penalty, save_checkpoint, show_tensor_images

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(epochs,
          dataloader: DataLoader,
          gen_model,
          gen_opt,
          crit_model,
          crit_opt,
          zdim,
          display_step,
          crit_cycles,
          gen_losses,
          crit_losses,
          cur_step,
          BATCH_SIZE,
          device=device):

    #training loop
    for epoch in range(epochs):
        for real, _ in dataloader:
            cur_bs = len(real)
            real = real.to(device)


            #critc training
            mean_crit_loss = 0
            crit_model.train()
            for _ in range(crit_cycles):
                crit_opt.zero_grad()

                noise = get_noise(cur_bs, zdim, device=device)
                fake = gen_model(noise)

                crit_fake_pred = crit_model(fake.detach())
                crit_real_pred = crit_model(real)

                alpha = torch.rand(len(real),1,1,1,device=device, requires_grad=True)
                gp = gradient_penalty(real, fake.detach(), crit_model, alpha)

                crit_loss = (torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + gp)
                mean_crit_loss = crit_loss.item()/crit_cycles

                crit_loss.backward(retain_graph=True)
                crit_opt.step()

            crit_losses += [mean_crit_loss]

            #gen training
            gen_model.train()
            gen_opt.zero_grad()

            noise = get_noise(BATCH_SIZE, zdim, device=device)
            fake = gen_model(noise)

            crit_fake_pred = crit_model(fake)
            gen_loss = -torch.mean(crit_fake_pred)
            gen_loss.backward()
            gen_opt.step()

            gen_losses += [gen_loss.item()]

            #show stats
            cur_step += 1

            if cur_step % 25 == 0 and cur_step > 0:
                mean_gen_loss = sum(gen_losses[-display_step:]) / display_step
                mean_crit_loss = sum(crit_losses[-display_step:]) / display_step
                print(f"Epoch: {epoch}, step: {cur_step}/{len(dataloader)*epochs} Generator loss: {mean_gen_loss:.4f}, critic loss: {mean_crit_loss:.4f}")
                save_checkpoint("model2", gen_model, gen_opt, crit_model, crit_opt, epoch)

            if cur_step % display_step == 0 and cur_step > 0:
                show_tensor_images(fake)
                show_tensor_images(real)
                step_bins = 5
                num_examples = (len(gen_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(gen_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss",
                )
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(crit_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss",
                )
                plt.xlabel("loss")
                plt.ylabel("batches")
                plt.legend()
                plt.grid(None)
                plt.show()
