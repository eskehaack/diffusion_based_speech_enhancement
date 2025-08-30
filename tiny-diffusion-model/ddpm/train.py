from pathlib import Path

import hydra
import numpy as np
import torch
from torch.utils.data import random_split
from hydra.utils import call
from tqdm import tqdm


@hydra.main(
    version_base=None,
    config_path="train_conf",
    config_name="sound_config",
)
def main(cfg) -> None:
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = call(cfg.dataset)

    if cfg.test.run_test:
        test_size = int(len(dataset) * cfg.test.test_split)
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(cfg.test.seed),
        )
        dataloader = call(cfg.dataloader, train_dataset)
        test_loader = call(cfg.dataloader, test_dataset)
    else:
        dataloader = call(cfg.dataloader, dataset)

    ns = call(cfg.noise_scheduler)
    model = call(cfg.model)
    model = model.to(device)
    optimizer = call(cfg.optimizer, model.parameters())
    criterion = torch.nn.MSELoss(reduction="mean")
    test_loss = 0

    # training
    with tqdm(total=cfg.num_epochs * len(dataloader)) as pbar:
        for _ in range(cfg.num_epochs):
            model.train()
            for batch in dataloader:
                batch = batch.to(device)
                t = torch.randint(0, ns.num_timesteps, (1,))[0].to(device)
                epsilon_target = torch.randn(batch.shape).to(device)  # noise
                x_t_plus_1 = ns.add_noise(batch, epsilon_target, t)  # x_{t+1}
                epsilon_pred = model(x_t_plus_1, t)  # predicted noise
                loss = criterion(epsilon_pred, epsilon_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                    {"test loss": f"{test_loss:.4f}", "batch_loss": f"{loss:.4f}"}
                )
                pbar.update(1)

            if cfg.test.run_test:
                test_loss = 0
                model.eval()
                for batch in test_loader:
                    batch = batch.to(device)
                    t = torch.randint(0, ns.num_timesteps, (1,))[0].to(device)
                    epsilon_target = torch.randn(batch.shape).to(device)  # noise
                    x_t_plus_1 = ns.add_noise(batch, epsilon_target, t)  # x_{t+1}

                    with torch.no_grad():
                        epsilon_pred = model(x_t_plus_1, t)  # predicted noise
                        loss = criterion(epsilon_pred, epsilon_target)

                    test_loss += loss.item()

                test_loss /= len(test_loader)

    torch.save(model.state_dict(), log_dir / "params.pt")
    print(f"saving model to {log_dir}")


if __name__ == "__main__":
    main()
