# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import yaml
import hydra
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from hydra.core.hydra_config import HydraConfig

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger



# @hydra.main(version_base=None, config_path="./conf/", config_name="train_config_cmu")
# @hydra.main(version_base=None, config_path="./conf/", config_name="train_config_cmu_v3")
# @hydra.main(version_base=None, config_path="./conf/", config_name="train_config_jrdb_t2p_v2")
# @hydra.main(version_base=None, config_path="./conf/", config_name="train_config_jrdb_t2p_v2_2_4")
# @hydra.main(version_base=None, config_path="./conf/", config_name="train_config_jrdb_t2p_v3_2_4")
# @hydra.main(version_base=None, config_path="./conf/", config_name="train_config_jrdb_t2p_v2_1_2_4_basic")
# @hydra.main(version_base=None, config_path="./conf/", config_name="train_config_3dpw_t2p_v3_0")
# @hydra.main(version_base=None, config_path="./conf/", config_name="train_config_jrdb_t2p_v3_0_2_4")
# @hydra.main(version_base=None, config_path="./conf/", config_name="train_config_jrdb_t2p_v2_3_6")
# @hydra.main(version_base=None, config_path="./conf/", config_name="train_config_jrdb_baseline")
# @hydra.main(version_base=None, config_path="./conf/", config_name="train_config_jrdb_baseline_2_4")
@hydra.main(version_base=None, config_path="./conf/", config_name="train_config_cmu_t2p_v3_finetune")
def main(conf):
    pl.seed_everything(conf.seed)
    output_dir = HydraConfig.get().runtime.output_dir

    model = instantiate(conf.model.target)
    model.output_dir = output_dir
    model.net = instantiate(conf.net.target)
    if conf.checkpoint is not None:
        print(f"Loading model from {conf.checkpoint}...")
        checkpoint = to_absolute_path(conf.checkpoint)
        assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"
    
        model_ckpt = torch.load(checkpoint)
        model.net.load_state_dict(model_ckpt['model'])
        # model = model.load_from_checkpoint(checkpoint)
    model.net.cuda()

    if conf.pretrained_weights is not None:
        ckpt_file = torch.load(conf.pretrained_weights)
        ckpt_state_dict = ckpt_file['state_dict']
        for key in list(ckpt_state_dict.keys()):
            ckpt_state_dict[key[4:]] = ckpt_state_dict[key]
            del ckpt_state_dict[key]
        model.net.load_state_dict(ckpt_state_dict)
    
    logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    model_checkpoint = ModelCheckpoint(dirpath=os.path.join(output_dir, "checkpoints"), monitor=conf.monitor, save_top_k=conf.save_top_k, mode='min')
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=conf.epochs,
        callbacks=[model_checkpoint],
        limit_val_batches=conf.limit_val_batches,
        limit_test_batches=conf.limit_test_batches,
    )

    # datamodule: pl.LightningDataModule = instantiate(conf.datamodule, test=conf.test)
    datamodule: pl.LightningDataModule = instantiate(conf.datamodule)
    datamodule.setup()
    
    print('Start training')
    # trainer.validate(model, datamodule.val_dataloader())
    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())
    ff = 1


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    main()


# from argparse import ArgumentParser

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint

# from datamodules import ArgoverseV1DataModule
# from models.hivt import HiVT

# if __name__ == '__main__':
#     pl.seed_everything(2022)

#     parser = ArgumentParser()
#     parser.add_argument('--root', type=str, required=True)
#     parser.add_argument('--train_batch_size', type=int, default=32)
#     parser.add_argument('--val_batch_size', type=int, default=32)
#     parser.add_argument('--shuffle', type=bool, default=True)
#     parser.add_argument('--num_workers', type=int, default=8)
#     parser.add_argument('--pin_memory', type=bool, default=True)
#     parser.add_argument('--persistent_workers', type=bool, default=True)
#     parser.add_argument('--gpus', type=int, default=1)
#     parser.add_argument('--max_epochs', type=int, default=64)
#     parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
#     parser.add_argument('--save_top_k', type=int, default=5)
#     parser = HiVT.add_model_specific_args(parser)
#     args = parser.parse_args()

#     model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
#     trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint])
#     model = HiVT(**vars(args))
#     datamodule = ArgoverseV1DataModule.from_argparse_args(args)
#     trainer.fit(model, datamodule)
