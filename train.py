import config
from ckpt_utils import load_checkpoint, save_checkpoint

import os
import time
import logging
import argparse
import numpy as np
from utils import *

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

IMG_RES = 224
IN_CHANNELS = 25
TL = 80
N_TRAJS = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--dev-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--img-res",
        type=int,
        required=False,
        default=IMG_RES,
        help="Input images resolution",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        required=False,
        default=IN_CHANNELS,
        help="Input raster channels",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        required=False,
        default=TL,
        help="Number time step to predict",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        required=False,
        default=N_TRAJS,
        help="Number of trajectories to predict",
    )
    parser.add_argument(
        "--save", type=str, required=True, help="Path to save model and logs"
    )

    parser.add_argument(
        "--model", type=str, required=False, default="xception71", help="CNN model name"
    )
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--batch-size", type=int, required=False, default=8)
    parser.add_argument("--n-epochs", type=int, required=False, default=60)

    parser.add_argument("--valid-limit", type=int, required=False, default=24 * 100)
    parser.add_argument(
        "--n-monitor-train",
        type=int,
        required=False,
        default=10,
        help="Validate model each `n-validate` steps",
    )
    parser.add_argument(
        "--n-monitor-validate",
        type=int,
        required=False,
        default=1000,
        help="Validate model each `n-validate` steps",
    )

    args = parser.parse_args()

    return args



class Model(nn.Module):
    def __init__(
        self, model_name, in_channels=IN_CHANNELS, time_limit=TL, n_traj=N_TRAJS
    ):
        super().__init__()

        self.n_traj = n_traj
        self.time_limit = time_limit
        f = 1
        while f: 
            try: 
                self.model = timm.create_model(
                    model_name,
                    pretrained=True,
                    in_chans=in_channels,
                    num_classes=self.n_traj * 2 * self.time_limit + self.n_traj,
                )
                f=0
                logging.info("Model Loaded !!!")
                print("Model Loaded !!!")
            except:
                logging.error("Loading model failed, trying again")
                print("Loading model failed, trying again")


    def forward(self, x):
        outputs = self.model(x)

        confidences_logits, logits = (
            outputs[:, : self.n_traj],
            outputs[:, self.n_traj :],
        )
        logits = logits.view(-1, self.n_traj, self.time_limit, 2)

        return confidences_logits, logits


def pytorch_neg_multi_log_likelihood_batch(gt, logits, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        logits (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(
        ((gt - logits) * avails) ** 2, dim=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )  # reduce time

    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=-1, keepdim=True)

    return torch.mean(error)


class WaymoLoader(Dataset):
    def __init__(self, directory, limit=0, return_vector=False, is_test=False):
        files = os.listdir(directory)
        self.files = [os.path.join(directory, f) for f in files if f.endswith(".npz")]

        if limit > 0:
            self.files = self.files[:limit]
        else:
            self.files = sorted(self.files)

        self.return_vector = return_vector
        self.is_test = is_test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        data = np.load(filename, allow_pickle=True)

        raster = data["raster"].astype("float32")
        raster = raster.transpose(2, 1, 0) / 255

        if self.is_test:
            center = data["shift"]
            yaw = data["yaw"]
            agent_id = data["object_id"]
            scenario_id = data["scenario_id"]

            return (
                raster,
                center,
                yaw,
                agent_id,
                str(scenario_id),
                data["_gt_marginal"],
                data["gt_marginal"],
            )

        trajectory = data["gt_marginal"]

        is_available = data["future_val_marginal"]

        if self.return_vector:
            return raster, trajectory, is_available, data["vector_data"]

        return raster, trajectory, is_available


def main():
    args = parse_args()
    os.makedirs(config.DIR.OUT_DIR, exist_ok=True)
    os.makedirs(config.DIR.CKPT_DIR, exist_ok=True)
    os.makedirs(config.DIR.TB_DIR, exist_ok=True)
    os.makedirs(config.DIR.VIS_DIR, exist_ok=True)

    # make csv best log file
    with open(os.path.join(config.DIR.OUT_DIR, "best_logs.csv"), "w") as f:
        f.write("epoch,loss\n")
        
    summary_writer = SummaryWriter(config.DIR.TB_DIR)
    logging.basicConfig(filename=os.path.join(config.DIR.OUT_DIR, 'train.log'), encoding='utf-8', level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')
    try: 
        model_name = config.MODEL_NAME
        time_limit = config.FUTURE_TS
        n_traj = config.N_TRAJ
        model = Model(
            model_name, in_channels=config.IN_CHANNELS, time_limit=time_limit, n_traj=n_traj
        )
    except Exception as e:
        logging.error("Loading model failed, trying again\n", str(e.__traceback__.tb_lineno)) 
        print("Loading model failed, trying again\n", str(e.__traceback__.tb_lineno))
    model = nn.DataParallel(model)
    model.cuda()

    lr = config.LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    if os.path.exists(os.path.join(config.DIR.CKPT_DIR, "last_model.pth")):
        config.LOAD_MODEL = True
    
    
    # Continue from last checkpoint
    if config.LOAD_MODEL:
        logging.info("Loading model from last checkpoint")
        end_epoch = load_checkpoint(os.path.join(config.DIR.CKPT_DIR, "last_model.pth"), model, optimizer)
    else : 
        logging.info("No checkpoint found, starting from scratch")
        end_epoch = 0
    
    if os.path.exists(os.path.join(config.DIR.OUT_DIR, "best_logs.csv")):
        with open(os.path.join(config.DIR.OUT_DIR, "best_logs.csv"), "r") as f:
            n = 0
            for line in f.readlines():
                n += 1
        if n > 1:
            best_loss = float(line.split(",")[1])
        else : 
            best_loss = 1e9
    else:
        best_loss = 1e9

    dataset = WaymoLoader(config.DIR.TRAIN_DIR)

    batch_size = config.TRAIN_BS
    num_workers = config.N_WORKERS
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=num_workers,
        pin_memory=False,
        # persistent_workers=True,
    )

    val_dataset = WaymoLoader(config.DIR.VAL_DIR, limit=args.valid_limit)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        # num_workers=num_workers,
        pin_memory=False,
        # persistent_workers=True,
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2 * len(dataloader),
        T_mult=1,
        eta_min=max(1e-2 * lr, 1e-6),
        last_epoch=-1,
    )

    glosses = []

    tr_it = iter(dataloader)
    n_epochs = config.EPOCHS
    # progress_bar_ex = tqdm(range(0, n_epochs * len(dataloader)), position=end_epoch*len(dataloader))
    progress_bar_ex = range(end_epoch*len(dataloader), n_epochs * len(dataloader))
    for iteration in progress_bar_ex:
        if iteration % config.LOG_STEP == 0:
            logging.info(f"iteration: {iteration} / {n_epochs * len(dataloader)}")

        model.train()
        try:
            x, y, is_available = next(tr_it)
        except StopIteration:
            tr_it = iter(dataloader)
            x, y, is_available = next(tr_it)

        x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

        optimizer.zero_grad()

        confidences_logits, logits = model(x)

        loss = pytorch_neg_multi_log_likelihood_batch(
            y, logits, confidences_logits, is_available
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        glosses.append(loss.item())
        if (iteration + 1) % args.n_monitor_train == 0:
            # progress_bar_ex.set_description(
            #     f"loss: {loss.item():.3}"
            #     f" avg: {np.mean(glosses[-100:]):.2}"
            #     f" {scheduler.get_last_lr()[-1]:.3}"
            # )
            logging.info(
                f"loss: {loss.item():.3}"
                f" avg: {np.mean(glosses[-100:]):.2}"
                f" {scheduler.get_last_lr()[-1]:.3}"
            )

            summary_writer.add_scalar("train/loss", loss.item(), iteration)
            summary_writer.add_scalar("lr", scheduler.get_last_lr()[-1], iteration)

        if (iteration + 1) % config.N_MONITOR_VALIDATE == 0:
            optimizer.zero_grad()
            model.eval()
            with torch.no_grad():
                val_losses = []
                for x, y, is_available in val_dataloader:
                    x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

                    confidences_logits, logits = model(x)
                    loss = pytorch_neg_multi_log_likelihood_batch(
                        y, logits, confidences_logits, is_available
                    )
                    val_losses.append(loss.item())

                summary_writer.add_scalar("dev/loss", np.mean(val_losses), iteration)
                t = time.localtime()

                current_time = time.strftime("%Y-%m-%d_%H-%M-%S", t)
                mean_val_loss = np.mean(val_losses)
                if mean_val_loss < best_loss:
                        best_loss = mean_val_loss
                        save_checkpoint(
                            config.DIR.CKPT_DIR,
                            model,
                            optimizer,
                            end_epoch,
                            date=current_time,
                            model_name=None,
                            name="model_best",
                        )
                        with open(os.path.join(config.DIR.OUT_DIR, "best_logs.csv"), "a") as f:
                            f.write(f"{end_epoch},{best_loss}\n")
                            
                        #model.eval()
                        #with torch.no_grad():
                            #traced_model = torch.jit.trace(
                                #model,
                                #torch.rand(
                     #               1, config.IN_CHANNELS, args.img_res, args.img_res
                      #          ).cuda(),
                      #      )

                    #    traced_model.save(os.path.join(config.DIR.OUT_DIR, "model_best.pt"))
                    #    del traced_model

        # End of Epoch                
        if (iteration+1) % len(dataloader) == 0:
            t = time.localtime()

            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", t)
            save_checkpoint(
                config.DIR.CKPT_DIR,
                model,
                optimizer,
                end_epoch,
                date=None,
                model_name=None,
                name="last_model",
            )
            
            save_checkpoint(
                config.DIR.CKPT_DIR,
                model,
                optimizer,
                end_epoch,
                date=current_time,
                model_name=None,
                name=f"model_{end_epoch}",
            )
            
            end_epoch += 1

if __name__ == "__main__":
    try: 
        main()
    except Exception as e:
        logging.error("Training failed\n", str(e.__traceback__.tb_lineno)) 
        print("Training failed\n", str(e.__traceback__.tb_lineno))
