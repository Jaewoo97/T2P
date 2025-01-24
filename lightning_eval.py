import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import yaml
import hydra
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from hydra.core.hydra_config import HydraConfig
import json

# @hydra.main(version_base=None, config_path="./conf/", config_name="eval_config_jrdb_baseline.yaml")
# @hydra.main(version_base=None, config_path="./conf/", config_name="eval_config_jrdb_baseline_2_4.yaml")
# @hydra.main(version_base=None, config_path="./conf/", config_name="eval_config_jrdb_t2p.yaml")
# @hydra.main(version_base=None, config_path="./conf/", config_name="eval_config_jrdb_t2p_2_4.yaml")
@hydra.main(version_base=None, config_path="./conf/", config_name="eval_config_cmu_t2p.yaml")
# @hydra.main(version_base=None, config_path="./conf/", config_name="eval_config_jrdb_baseline_1_2_frameRep15.yaml")
# @hydra.main(version_base=None, config_path="./conf/", config_name="eval_config_cmu_t2p.yaml")
# @hydra.main(version_base=None, config_path="./conf/", config_name="eval_config_cmu_baseline.yaml")
def main(conf):
    pl.seed_everything(conf.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    checkpoint = to_absolute_path(conf.checkpoint)
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    model = instantiate(conf.model.target)
    model.net = instantiate(conf.net.target)
    model.output_dir = output_dir
    
    if checkpoint[-5:] == '.ckpt':
        ckpt_file = torch.load(checkpoint)
        ckpt_state_dict = ckpt_file['state_dict']
        for key in list(ckpt_state_dict.keys()):
            ckpt_state_dict[key[4:]] = ckpt_state_dict[key]
            del ckpt_state_dict[key]
        model.net.load_state_dict(ckpt_state_dict)
        print('Model loaded!')
        # model = model.load_from_checkpoint(checkpoint)
    else:
        model_ckpt = torch.load(checkpoint)
        model.net.load_state_dict(model_ckpt['model'])
        print('Model loaded!')
        
    model.val_metrics.reset()

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=1,
        limit_val_batches=conf.limit_val_batches,
        limit_test_batches=conf.limit_test_batches,
    )

    # datamodule: pl.LightningDataModule = instantiate(conf.datamodule, test=conf.test)
    datamodule: pl.LightningDataModule = instantiate(conf.datamodule)
    datamodule.setup()
    
    print('Start validation')
    val_results = trainer.validate(model, datamodule.val_dataloader())
    with open(output_dir+'\eval_dir.json','w') as f:
        json.dump(val_results[0], f, indent=4)
    ff = 1


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    main()