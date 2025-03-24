from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.metrics_computer import compute_coverage_indicator, compute_log_region_size
from moc.conformal.conformalizers import L_CP


config = get_config()
config.device = 'cpu'
rc = RunConfig(config, 'mulan', 'sf2')
datamodule = RealDataModule(rc)
p, q = datamodule.input_dim, datamodule.output_dim
model = MQF2LightningModule(p, q)
trainer = get_lightning_trainer(rc)
trainer.fit(model, datamodule)

alpha = 0.1
conformalizer = L_CP(datamodule.calib_dataloader(), model)
test_batch = next(iter(datamodule.test_dataloader()))
x, y = test_batch
coverage = compute_coverage_indicator(conformalizer, alpha, x, y)
volume = compute_log_region_size(conformalizer, model, alpha, x, n_samples=100)
print(coverage)
print(volume)