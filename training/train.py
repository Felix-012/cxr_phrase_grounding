"""adapted from https://github.com/huggingface/diffusers/tree/main/examples/text_to_image"""

import argparse
import logging
from datetime import timedelta
import torchvision.transforms.functional as functional
import accelerate
from transformers import CLIPTokenizer, CLIPTextModel

from custom_pipe import _load_unet
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
from diffusers import UNet2DConditionModel, EMAModel, AutoencoderKL, DDIMScheduler
from packaging import version
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from custom_pipe import FrozenCustomPipe
from datasets import get_dataset
from datasets.utils import load_config
from util_scripts.attention_maps import cross_attn_init, set_layer_with_name_and_path, register_cross_attention_hook, \
    all_attn_maps, all_neg_attn_maps
from util_scripts.preliminary_masks import preprocess_attention_maps
from util_scripts.utils_generic import collate_batch
from util_scripts.utils_train import get_latest_directory, get_parser_arguments_train, tokenize_captions, unwrap_model, \
    normalize_and_scale_tensor
from evaluation.compute_bbox import vis

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser()
    parser = get_parser_arguments_train(parser)
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
    from diffusers.training_utils import compute_snr
    from diffusers.utils import check_min_version
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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler, tokenizer and models.
    if args.use_custom:
        pipeline = FrozenCustomPipe(path=args.pretrained_model_name_or_path, accelerator=accelerator, llm_name=args.llm_name)
        unet = pipeline.pipe.unet
        vae = pipeline.pipe.vae
        text_encoder = pipeline.pipe.text_encoder
        tokenizer = pipeline.pipe.tokenizer
        noise_scheduler = pipeline.pipe.scheduler
    else:
        noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, torch_dtype=weight_dtype
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, torch_dtype=weight_dtype
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant, torch_dtype=torch.float32
        )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder = text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

    unet.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = _load_unet(component_name="unet", path=args.pretrained_model_name_or_path, torch_dtype=torch.float32)
        ema_unet.to("cuda")
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

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

        # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to("cuda")
                ema_unet.copy_to(unet.parameters())
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

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
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    config = load_config(args.config)
    with accelerator.main_process_first():
        train_dataset = get_dataset(config, "train")
        if config.num_chunks > 1:
            accelerator.print("using chunked data")
            train_dataset.load_next_chunk()
        else:
            accelerator.print("using whole dataset")
            train_dataset.load_precomputed(vae)


        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_batch,
            batch_size=args.train_batch_size,
            num_workers=config.dataloading.num_workers,
        )
        accelerator.print("Tokenizing training data...")
        for data in train_dataset:
            impression = data['impression'] if 'impression' in data else None
            if impression:
                data['input_ids'], data['attention_mask'] = tokenize_captions([impression], tokenizer,
                                                                              is_train=True)
            else:
                raise KeyError("No impression saved")


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
    unet, optimizer, lr_scheduler, noise_scheduler, vae, text_encoder = accelerator.prepare(
        unet, optimizer, lr_scheduler, noise_scheduler, vae, text_encoder,
        device_placement=[True, True, True, True, True, True]
    )

    accelerator.register_for_checkpointing(lr_scheduler)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
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

    empty_token_id = tokenizer(
        "", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        with accelerator.main_process_first():
            with accelerator.autocast():
                if config.num_chunks > 1 and epoch != first_epoch:
                    train_dataset.load_next_chunk()
                    train_dataloader = DataLoader(
                        train_dataset,
                        shuffle=True,
                        collate_fn=collate_batch,
                        batch_size=args.train_batch_size,
                        num_workers=config.dataloading.num_workers,
                    )
                    accelerator.print("Tokenizing training data...")
                    for data in train_dataset:
                        impression = data['impression'] if 'impression' in data else None
                        if impression:
                            data['input_ids'], data['attention_mask'] = tokenize_captions([impression], tokenizer,
                                                                                          is_train=True)
                        else:
                            raise KeyError("No impression saved")

            for name, param in unet.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN detected in {name} before training starts.")
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    for i in range(len(batch["input_ids"])):
                        if bool(torch.rand(1) < args.ucg_probability):
                            batch["input_ids"][i] = empty_token_id
                    # Convert images to latent space
                    latents = torch.cat([latent.latent_dist.sample() for latent in batch["img"]]).to(device=unet.device, dtype=weight_dtype )
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
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

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
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_unet.step(unet.parameters())
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
                            unet.save_pretrained(save_path)

                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

                if accelerator.is_main_process:
                    if args.validation_prompt is not None and progress_bar.n % args.generation_validation_epochs == 0:
                        try:
                            get_latest_directory(args)
                        except TypeError:
                            logger.info(f"Skipping validation - checkpoint {args.resume_from_checkpoint} could not be found")
                            continue
                        logger.info(
                            f"Running validation generation... \n Generating {args.num_validation_images} images with prompt:"
                            f" {args.validation_prompt}."
                        )

                        pipeline = StableDiffusionPipeline(
                                vae=accelerator.unwrap_model(vae),
                                text_encoder=accelerator.unwrap_model(text_encoder),
                                tokenizer=tokenizer,
                                unet=accelerator.unwrap_model(unet),
                                safety_checker=None,
                                feature_extractor=None,
                                scheduler=noise_scheduler
                        )
                        cross_attn_init()
                        pipeline.unet = set_layer_with_name_and_path(pipeline.unet)
                        pipeline.unet = register_cross_attention_hook(pipeline.unet)


                        pipeline = pipeline
                        pipeline.set_progress_bar_config(disable=True)

                        if args.enable_xformers_memory_efficient_attention:
                            pipeline.enable_xformers_memory_efficient_attention()

                        # run inference
                        generator = torch.Generator(device="cuda")
                        if args.seed is not None:
                            generator = generator.manual_seed(args.seed)
                        if torch.backends.mps.is_available():
                            autocast_ctx = nullcontext()
                        else:
                            autocast_ctx = torch.autocast(accelerator.device.type)

                        images = []
                        attention_images = []
                        for i in range(args.num_validation_images):
                            all_neg_attn_maps.clear()
                            all_attn_maps.clear()

                            with autocast_ctx:
                                image = pipeline(args.validation_prompt, num_inference_steps=20,
                                                 generator=generator).images[0]
                            images.append(image)

                            attention_images.append(preprocess_attention_maps(all_attn_maps, on_cpu=True))


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
                                        ],
                                        "attention": [
                                            wandb.Image(functional.to_pil_image(
                                                normalize_and_scale_tensor(image.squeeze()[:,:,1:-1].mean(dim=(0,1,2)))), caption=f"{i}: {args.validation_prompt}")
                                            for i, image in enumerate(attention_images)
                                        ]
                                    }
                                )

                        del pipeline
                torch.cuda.empty_cache()

    # Save the layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet, accelerator)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )

        pipeline.unet.save_pretrained(args.output_dir)

        accelerator.end_training()





if __name__ == "__main__":
    main()
