from ray.train.huggingface import HuggingFaceTrainer
from ray.air.config import ScalingConfig
from ray.data.preprocessors import Chain
from ray.data.preprocessors import BatchMapper
from dataset_handler import split_text,tokenize
from config import trainer_init_per_worker


os.environ["COMET_API_KEY"]= "cHvndCQ5jOPyohd4g2x80eA43"

model_name = "bigscience/bloom-1b1"
use_gpu = True
num_workers = 2
cpus_per_worker = 8
block_size = 256


splitter = BatchMapper(split_text, batch_format="pandas")
tokenizer = BatchMapper(tokenize, batch_format="pandas")


trainer = HuggingFaceTrainer(
    trainer_init_per_worker=trainer_init_per_worker,
    trainer_init_config={
        "batch_size": 8,  # per device
        "epochs": 1,
    },
    scaling_config=ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"GPU": 1, "CPU": 8},
    ),
    datasets={"train": ray_datasets["train"], "evaluation": ray_datasets["validation"]},
    preprocessor=Chain(splitter, tokenizer),
)



results = trainer.fit()

checkpoint = results.checkpoint
checkpoint
