# parameters
result_dir="/home/eric/temp/SeViLA/result"

exp_name='star_sr'
ckpt='/home/eric/temp/SeViLA/sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node=1 train.py \
--cfg-path lavis/projects/sevila/train/star.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.star.vis_processor.train.n_frms=4 \
datasets.star.vis_processor.eval.n_frms=32 \
run.batch_size_train=16 \
run.batch_size_eval=12 \
run.init_lr=3e-5 \
run.max_epoch=10 \
run.warmup_steps=500 \
run.accum_grad_iters=1 \
model.task='train_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'