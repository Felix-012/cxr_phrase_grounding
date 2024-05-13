"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
from datetime import timedelta

import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from radbert_pipe import FrozenRadBERTPipe
from datasets import get_dataset
from datasets.utils import load_config
from utils.utils import collate_batch
from utils.utils_train import get_latest_directory, get_parser_arguments_train_lora, tokenize_captions, unwrap_model
from sklearn.model_selection import train_test_split

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser()
    parser = get_parser_arguments_train_lora(parser)
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main():
    args = parse_args()

    os.environ['HF_HOME'] = args.cache_dir
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import diffusers
    from diffusers import StableDiffusionPipeline
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params, compute_snr
    from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
    from diffusers.utils.import_utils import is_xformers_available

    # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
    check_min_version("0.28.0.dev0")

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir))
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=8000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.
    pipeline = FrozenRadBERTPipe(path=args.pretrained_model_name_or_path, accelerator=accelerator)
    unet = pipeline.pipe.unet
    vae = pipeline.pipe.vae
    text_encoder = pipeline.pipe.text_encoder
    tokenizer = pipeline.pipe.tokenizer
    noise_scheduler = pipeline.pipe.scheduler

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)


    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    config = load_config(args.config)
    with accelerator.main_process_first():

        dataset = get_dataset(config, "train")

        # Splitting dataset into train and validation subsets
        train_idx, val_idx = train_test_split(
            range(len(dataset)),
            test_size=config.validation_split,
            random_state=42
        )

        if hasattr(dataset, 'load_precomputed'):
            dataset.load_precomputed(pipeline.pipe.vae)

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)



        # Tokenizing data
        accelerator.print("Tokenizing training data...")
        for data in train_dataset:
            impression = data['impression'] if 'impression' in data else None
            if impression:
                data['input_ids'], data['attention_mask'] = tokenize_captions([impression], tokenizer, is_train=True)
            else:
                raise KeyError("No impression saved")

        for data in val_dataset:
            impression = data['impression'] if 'impression' in data else None
            if impression:
                data['input_ids'], data['attention_mask'] = tokenize_captions([impression], tokenizer, is_train=True)
            else:
                raise KeyError("No impression saved")



        # Create dataloaders for training and validation datasets
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_batch,
            batch_size=args.train_batch_size,
            num_workers=config.dataloading.num_workers,
        )

        val_dataloader = DataLoader(
            val_dataset,
            shuffle=True,
            collate_fn=collate_batch,
            batch_size=args.train_batch_size,
            num_workers=config.dataloading.num_workers,
        )




    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler, noise_scheduler, vae, text_encoder = accelerator.prepare(
        [unet, optimizer, train_dataloader, lr_scheduler, noise_scheduler, vae, text_encoder]
    )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = get_latest_directory(args)
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = torch.cat([latent.latent_dist.sample() for latent in batch["img"]]).to(unet.device)
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=unet.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=unet.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=weight_dtype)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False, attention_mask=batch["attention_mask"])[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = unwrap_model(unet, accelerator)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


        if accelerator.is_main_process:
            if progress_bar.n % args.validation_epochs == 0:
                logger.info("Running validation...")

                unet.eval()  # Set the model to evaluation mode
                validation_loss = 0.0
                num_batches = 0

                with torch.no_grad():  # No gradients needed for validation
                    for val_batch in val_dataloader:
                        # Convert images to latent space
                        latents = torch.cat([latent.latent_dist.sample() for latent in val_batch["img"]])
                        latents = latents * vae.config.scaling_factor

                        # No noise is added to the latents in validation
                        bsz = latents.shape[0]
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                                  device="cuda")
                        timesteps = timesteps.long()

                        # Get the text embedding for conditioning
                        encoder_hidden_states = text_encoder(val_batch["input_ids"], return_dict=False, attention_mask=val_batch["attention_mask"])[0]

                        # Compute noisy latents by using true noise during validation for consistency
                        noise = torch.randn_like(latents)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        # Predict the noise residual
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                        # Calculate the loss
                        val_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        validation_loss += val_loss.item()
                        num_batches += 1

                avg_validation_loss = validation_loss / num_batches
                accelerator.log({"validation_loss": avg_validation_loss}, step=global_step)

                logger.info(f"Validation completed: Avg Loss = {avg_validation_loss}")
        if accelerator.is_main_process:
            if args.validation_prompt is not None and progress_bar.n % args.generation_validation_epochs == 0:
                logger.info(
                    f"Running validation generation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                pipeline = FrozenRadBERTPipe(path=args.pretrained_model_name_or_path).pipe
                pipeline = pipeline.to("cuda")
                pipeline.load_lora_weights(os.path.join(args.output_dir, get_latest_directory(args)))
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device="cuda")
                if args.seed is not None:
                    generator = generator.manual_seed(args.seed)
                images = []
                if torch.backends.mps.is_available():
                    autocast_ctx = nullcontext()
                else:
                    autocast_ctx = torch.autocast(accelerator.device.type)

                with autocast_ctx:
                    for _ in range(args.num_validation_images):
                        images.append(
                            pipeline(args.validation_prompt, num_inference_steps=30,
                                     generator=generator).images[0]
                        )

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del pipeline
        torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet, accelerator)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        accelerator.end_training()





if __name__ == "__main__":
    main()
