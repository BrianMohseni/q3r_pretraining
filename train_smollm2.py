import argparse
import contextlib
import math
import shutil
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from q3r.adamq3r import AdamQ3R


CHECKPOINT_OPTIMIZERS = ("adamw", "adamq3r")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain HuggingFaceTB/SmolLM2-135M on streamed FineWeb with "
            "AdamW or AdamQ3R using Hugging Face Accelerate."
        )
    )
    parser.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--reinitialize_weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset_config", default="sample-10BT")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", default="checkpoints/smollm2-135m")
    parser.add_argument("--keep_last_checkpoints", type=int, default=2)
    parser.add_argument("--optimizer", choices=["adamw", "adamq3r"], default="adamq3r")
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_learning_rate_ratio", type=float, default=0.10)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--num_steps", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1_000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--wandb_project", default="adamq3r")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
        help=(
            "Accelerate mixed-precision mode. The default preserves the original "
            "bf16 training behavior."
        ),
    )
    parser.add_argument(
        "--attention_implementation",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help=(
            "Attention backend passed to Transformers. 'sdpa' uses PyTorch scaled "
            "dot-product attention, which dispatches to flash kernels when available."
        ),
    )
    parser.add_argument(
        "--sdp_kernel",
        choices=["auto", "flash_only", "efficient_only", "math_only"],
        default="auto",
        help="CUDA scaled-dot-product attention kernel policy.",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compile_mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument("--compile_fullgraph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--matmul_precision",
        choices=["highest", "high", "medium"],
        default="high",
        help="Float32 matmul precision. 'high' enables TF32 acceleration on Ampere+ GPUs.",
    )
    parser.add_argument(
        "--adamw_impl",
        choices=["auto", "fused", "foreach", "default"],
        default="auto",
        help="AdamW backend used when --optimizer adamw.",
    )
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--q3r_steps", type=int, default=50)
    parser.add_argument("--q3r_lambda", type=float, default=2e-4)
    parser.add_argument("--q3r_target_rank", type=float, default=0.125)
    parser.add_argument("--q3r_verbose", action="store_true")
    args = parser.parse_args()

    if args.warmup_steps >= args.num_steps:
        parser.error("--warmup_steps must be smaller than --num_steps.")
    if not 0.0 <= args.min_learning_rate_ratio <= 1.0:
        parser.error("--min_learning_rate_ratio must be between 0.0 and 1.0.")
    if args.keep_last_checkpoints < 1:
        parser.error("--keep_last_checkpoints must be at least 1.")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient_accumulation_steps must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")
    if args.log_every < 1:
        parser.error("--log_every must be at least 1.")
    return args


def format_token_count(num_tokens):
    units = ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M"))
    for divisor, suffix in units:
        if num_tokens >= divisor or suffix == "M":
            value = num_tokens / divisor
            return f"{value:.2f}".rstrip("0").rstrip(".") + suffix


def training_token_count(args, steps=None, world_size=1):
    if steps is None:
        steps = args.num_steps
    return (
        steps
        * args.batch_size
        * args.gradient_accumulation_steps
        * args.context_length
        * world_size
    )


def build_wandb_run_name(args, world_size):
    num_training_tokens = format_token_count(
        training_token_count(args, world_size=world_size)
    )
    return f"{args.model_name}_{args.optimizer}_{num_training_tokens}"


def init_trackers(args, accelerator):
    if args.no_wandb:
        return

    total_training_tokens = training_token_count(
        args,
        world_size=accelerator.num_processes,
    )
    run_name = build_wandb_run_name(args, accelerator.num_processes)
    config = {
        **vars(args),
        "num_processes": accelerator.num_processes,
        "distributed_type": str(accelerator.distributed_type),
        "num_training_tokens": total_training_tokens,
        "num_training_tokens_formatted": format_token_count(total_training_tokens),
        "wandb_run_name": run_name,
    }

    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=config,
        init_kwargs={"wandb": {"name": run_name}},
    )


def build_dataset(args, tokenizer):
    if args.train_file:
        suffix = Path(args.train_file).suffix.lower()
        if suffix == ".txt":
            dataset = load_dataset("text", data_files={"train": args.train_file}, split="train")
        elif suffix in {".json", ".jsonl"}:
            dataset = load_dataset("json", data_files={"train": args.train_file}, split="train")
        else:
            raise ValueError("--train_file must be .txt, .json, or .jsonl")
    else:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.train_split,
            streaming=args.streaming,
        )

    context_length = args.context_length

    def tokenize(batch):
        texts = batch[args.text_column]
        return tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_attention_mask=True,
        )

    remove_columns = getattr(dataset, "column_names", None) or [args.text_column]
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=remove_columns,
    )
    tokenized = tokenized.with_format("torch")
    return tokenized


def collate_batch(features):
    input_ids = torch.stack([feature["input_ids"] for feature in features])
    attention_mask = torch.stack([feature["attention_mask"] for feature in features])
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_model(args):
    model_kwargs = {}
    if args.attention_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attention_implementation

    if args.reinitialize_weights:
        config = AutoConfig.from_pretrained(args.model_name)
        config.use_cache = False
        model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        model.config.use_cache = False

    return model


def configure_torch_runtime(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    torch.set_float32_matmul_precision(args.matmul_precision)

    if not torch.cuda.is_available():
        return

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        if args.sdp_kernel == "auto":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        elif args.sdp_kernel == "flash_only":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
        elif args.sdp_kernel == "efficient_only":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        elif args.sdp_kernel == "math_only":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)


def compile_model(args, model):
    if not args.compile:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("--compile requires torch.compile, but this PyTorch build does not provide it.")

    return torch.compile(
        model,
        mode=args.compile_mode,
        fullgraph=args.compile_fullgraph,
        dynamic=False,
    )


def mark_cudagraph_step_begin():
    compiler = getattr(torch, "compiler", None)
    mark_step_begin = getattr(compiler, "cudagraph_mark_step_begin", None)
    if mark_step_begin is not None:
        mark_step_begin()


def q3r_trainable_modules(model):
    embedding_weight_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
    }

    return {
        module: None
        for name, module in model.named_modules()
        if (
            isinstance(module, nn.Linear)
            and module.weight.requires_grad
            and name.split(".")[-1] != "lm_head"
            and id(module.weight) not in embedding_weight_ids
        )
    }


def build_optimizer(args, model):
    if args.optimizer == "adamw":
        optimizer_kwargs = {
            "lr": args.learning_rate,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_eps,
            "weight_decay": 1e-3,
        }
        if args.adamw_impl in {"auto", "fused"} and torch.cuda.is_available():
            optimizer_kwargs["fused"] = True
        elif args.adamw_impl == "foreach":
            optimizer_kwargs["foreach"] = True

        try:
            return AdamW(model.parameters(), **optimizer_kwargs)
        except (RuntimeError, TypeError):
            if args.adamw_impl != "auto":
                raise
            optimizer_kwargs.pop("fused", None)
            optimizer_kwargs["foreach"] = True
            with contextlib.suppress(RuntimeError, TypeError):
                return AdamW(model.parameters(), **optimizer_kwargs)
            optimizer_kwargs.pop("foreach", None)
            return AdamW(model.parameters(), **optimizer_kwargs)

    trainable_modules = q3r_trainable_modules(model)
    if not trainable_modules:
        raise ValueError("AdamQ3R requires at least one trainable nn.Linear module.")
    return AdamQ3R(
        model.parameters(),
        trainable_modules=trainable_modules,
        target_rank=args.q3r_target_rank,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        lmbda=args.q3r_lambda,
        N=args.num_steps,
        steps=args.q3r_steps,
        verbose=args.q3r_verbose,
    )


def build_scheduler(args, optimizer):
    min_lr_ratio = args.min_learning_rate_ratio
    decay_steps = args.num_steps - args.warmup_steps

    def lr_lambda(step):
        if step < args.warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * (step / args.warmup_steps)
        decay_progress = min((step - args.warmup_steps) / decay_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def checkpoint_root(args, optimizer=None):
    return Path(args.output_dir) / "model" / (optimizer or args.optimizer)


def checkpoint_step(path):
    if not path.name.startswith("step-"):
        return None
    with contextlib.suppress(ValueError):
        return int(path.name.removeprefix("step-"))
    return None


def ensure_checkpoint_layout(args):
    for optimizer in CHECKPOINT_OPTIMIZERS:
        checkpoint_root(args, optimizer).mkdir(parents=True, exist_ok=True)


def prune_checkpoints(args, optimizer=None):
    if args.keep_last_checkpoints <= 0:
        return

    root = checkpoint_root(args, optimizer)
    checkpoints = []
    for path in root.glob("step-*"):
        if not path.is_dir():
            continue
        step = checkpoint_step(path)
        if step is not None:
            checkpoints.append((step, path))

    checkpoints.sort()
    for _, path in checkpoints[:-args.keep_last_checkpoints]:
        shutil.rmtree(path)


def save_checkpoint(
    accelerator,
    model,
    tokenizer,
    optimizer,
    scheduler,
    args,
    step,
):
    checkpoint_dir = checkpoint_root(args) / f"step-{step}"

    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(
        model,
        keep_torch_compile=False,
    )
    state_dict = accelerator.get_state_dict(model)
    unwrapped_model.save_pretrained(
        checkpoint_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
    )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_dir)
        accelerator.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "args": vars(args),
                "num_processes": accelerator.num_processes,
                "distributed_type": str(accelerator.distributed_type),
            },
            checkpoint_dir / "training_state.pt",
        )
        prune_checkpoints(args)

    accelerator.wait_for_everyone()


def next_batch(dataloader, dataloader_iter):
    try:
        return next(dataloader_iter), dataloader_iter
    except StopIteration:
        dataloader_iter = iter(dataloader)
        return next(dataloader_iter), dataloader_iter


def main():
    args = parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None if args.no_wandb else "wandb",
        project_dir=args.output_dir,
    )

    torch.manual_seed(args.seed)
    configure_torch_runtime(args)

    if accelerator.device.type != "cuda":
        raise RuntimeError("This training script requires a CUDA device.")
    if args.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not report bf16 support.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(args)
    model.train()

    dataset = build_dataset(args, tokenizer)
    dataloader_num_workers = 0 if args.streaming else args.num_workers
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_batch,
        "num_workers": dataloader_num_workers,
        "pin_memory": True,
    }
    if dataloader_num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader_kwargs["persistent_workers"] = args.persistent_workers
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    model = compile_model(args, model)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler,
    )
    model.train()

    if accelerator.is_main_process:
        ensure_checkpoint_layout(args)
        for optimizer_name in CHECKPOINT_OPTIMIZERS:
            prune_checkpoints(args, optimizer_name)
    accelerator.wait_for_everyone()

    init_trackers(args, accelerator)

    optimizer.zero_grad(set_to_none=True)
    running_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    running_loss_count = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    completed_steps = 0
    dataloader_iter = iter(dataloader)
    last_log_time = time.perf_counter()
    last_log_tokens = 0
    real_tokens_seen_local = 0

    try:
        while completed_steps < args.num_steps:
            batch, dataloader_iter = next_batch(dataloader, dataloader_iter)

            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = input_ids.masked_fill(attention_mask == 0, -100)
                real_tokens_seen_local += int(attention_mask.sum().item())

                if args.compile:
                    mark_cudagraph_step_begin()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss_sum += loss.detach().float()
            running_loss_count += 1

            if not accelerator.sync_gradients:
                continue

            completed_steps += 1

            if completed_steps % args.log_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.device.type == "cuda":
                    torch.cuda.synchronize(accelerator.device)

                now = time.perf_counter()
                reduced_loss_sum = accelerator.reduce(
                    running_loss_sum,
                    reduction="sum",
                )
                reduced_loss_count = accelerator.reduce(
                    running_loss_count,
                    reduction="sum",
                )
                global_tokens_seen_tensor = torch.tensor(
                    real_tokens_seen_local,
                    device=accelerator.device,
                    dtype=torch.long,
                )
                global_tokens_seen = int(
                    accelerator.reduce(
                        global_tokens_seen_tensor,
                        reduction="sum",
                    ).item()
                )

                avg_loss = (reduced_loss_sum / reduced_loss_count.clamp_min(1)).item()
                perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                lr = scheduler.get_last_lr()[0]
                elapsed = now - last_log_time
                tokens_per_second = (
                    (global_tokens_seen - last_log_tokens) / elapsed
                    if elapsed > 0
                    else float("inf")
                )

                accelerator.print(
                    f"step={completed_steps} "
                    f"loss={avg_loss:.4f} "
                    f"ppl={perplexity:.2f} "
                    f"lr={lr:.6g} "
                    f"tok/s={tokens_per_second:.2f}"
                )

                if not args.no_wandb:
                    accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/perplexity": perplexity,
                            "train/learning_rate": lr,
                            "train/tokens": global_tokens_seen,
                            "train/tokens_per_second": tokens_per_second,
                        },
                        step=completed_steps,
                    )

                running_loss_sum.zero_()
                running_loss_count.zero_()
                last_log_time = now
                last_log_tokens = global_tokens_seen

            if args.save_every and completed_steps % args.save_every == 0:
                save_checkpoint(
                    accelerator,
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    args,
                    completed_steps,
                )

        save_checkpoint(
            accelerator,
            model,
            tokenizer,
            optimizer,
            scheduler,
            args,
            completed_steps,
        )
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
