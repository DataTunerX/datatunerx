import json
import logging
from dataclasses import field, dataclass
from typing import Optional, Dict, Any, Tuple, Literal

import transformers
from transformers import Seq2SeqTrainingArguments, HfArgumentParser

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co."}
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    split_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Will use the token generated when running `huggingface-cli login`."}
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )
    quantization_type: Optional[Literal["fp4", "nf4"]] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."}
    )
    double_quantization: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use double quantization in int4 training or not."}
    )
    quantization: Optional[str] = field(
        default=None,
        metadata={"help": "quantize the model, int4, int8, or None."}
    )

    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Adopt scaled rotary positional embeddings."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory(s) containing the delta model checkpoints as well as the configurations."}
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    shift_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."}
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the reward model."}
    )
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to plot the training loss after fine-tuning or not."}
    )
    hf_auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."}
    )
    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."}
    )

    def __post_init__(self):
        self.compute_dtype = None
        self.model_max_length = None

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.checkpoint_dir is not None:  # support merging multiple lora weights
            self.checkpoint_dir = [cd.strip() for cd in self.checkpoint_dir.split(",")]

        if self.quantization_bit is not None:
            assert self.quantization_bit in [4, 8], "We only accept 4-bit or 8-bit quantization."

        if self.quantization is not None:
            assert self.quantization in ["int4", "int8"], "We only accept int4 or int8 quantization."

        if self.use_auth_token == True and self.hf_auth_token is not None:
            from huggingface_hub.hf_api import HfFolder  # lazy load
            HfFolder.save_token(self.hf_auth_token)


@dataclass
class FinetuningArguments:
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    stage: Optional[Literal["pt", "sft", "rm", "ppo", "dpo"]] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."}
    )
    finetuning_type: Optional[Literal["lora", "freeze", "full", "none"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."}
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for partial-parameter (freeze) fine-tuning."}
    )
    name_module_trainable: Optional[Literal["mlp", "self_attn", "self_attention"]] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for partial-parameter (freeze) fine-tuning. \
                  LLaMA choices: [\"mlp\", \"self_attn\"], \
                  BLOOM & Falcon & ChatGLM2 choices: [\"mlp\", \"self_attention\"], \
                  Qwen choices: [\"mlp\", \"attn\"], \
                  Phi-1.5 choices: [\"mlp\", \"mixer\"], \
                  LLaMA-2, Baichuan, InternLM, XVERSE choices: the same as LLaMA."}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "The scale factor for LoRA fine-tuning (similar with the learning rate)."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )
    lora_target: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  BLOOM & Falcon & ChatGLM2 choices: [\"query_key_value\", \"self_attention.dense\", \"mlp.dense\"], \
                  Baichuan choices: [\"W_pack\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], \
                  Qwen choices: [\"c_attn\", \"attn.c_proj\", \"w1\", \"w2\", \"mlp.c_proj\"], \
                  Phi-1.5 choices: [\"Wqkv\", \"out_proj\", \"fc1\", \"fc2\"], \
                  LLaMA-2, InternLM, XVERSE choices: the same as LLaMA."}
    )
    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."}
    )
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to resume training from the last LoRA weights or create new weights after merging them."}
    )
    ppo_score_norm: Optional[bool] = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."}
    )
    ppo_logger: Optional[str] = field(
        default=None,
        metadata={"help": "Log with either 'wandb' or 'tensorboard' in PPO training."}
    )
    ppo_target: Optional[float] = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."}
    )
    dpo_beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta parameter for the DPO loss."}
    )
    upcast_layernorm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to upcast the layernorm weights in fp32."}
    )
    neft_alpha: Optional[float] = field(
        default=0,
        metadata={"help": "The alpha parameter to control the noise magnitude in NEFTune."}
    )
    num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "Number of worker."}
    )
    storage_path: Optional[str] = field(
        default=None,
        metadata={"help": "storage_path is used to storage checkpoint."}
    )
    metrics_export_address: Optional[str] = field(
        default=None,
        metadata={"help": "address to export train metrics."}
    )
    uid: Optional[str] = field(
        default=None,
        metadata={"help": "finetune crd uid."}
    )

    def __post_init__(self):
        if isinstance(self.lora_target, str):  # support custom target modules/layers of LoRA
            self.lora_target = [target.strip() for target in self.lora_target.split(",")]

        if isinstance(self.additional_target, str):
            self.additional_target = [target.strip() for target in self.additional_target.split(",")]

        assert self.finetuning_type in ["lora", "freeze", "full", "none"], "Invalid fine-tuning method."

        if not self.storage_path:
            raise ValueError("--storage_path must be specified")


@dataclass
class DataArguments:
    train_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to train dataset"}
    )

    evaluation_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation dataset"}
    )

    columns: Optional[str] = field(
        default=None,
        metadata={"help": "columns map for dataset"}
    )
    block_size: Optional[int] = field(
        default=1024,
        metadata={"help": "length of input."}
    )

    def __post_init__(self):
        if self.train_path is None:
            raise ValueError("--train_path must be specified")


def get_train_args() -> Tuple[Seq2SeqTrainingArguments, FinetuningArguments, ModelArguments, DataArguments]:
    parser = HfArgumentParser((Seq2SeqTrainingArguments, FinetuningArguments, ModelArguments, DataArguments))

    training_args, finetuning_args, model_args, data_args = parser.parse_args_into_dataclasses()

    training_args.optim = "adamw_torch" if training_args.optim == "adamw_hf" else training_args.optim  # suppress warning

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"  distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    return training_args, finetuning_args, model_args, data_args
