# parameters/data path
result_dir="/home/eric/temp/SeViLA-test/result/"

# exp_name='star_ft_infer'
exp_name='star_ft_infer_0shot_pdcv_time_test'

ckpt='/home/eric/temp/SeViLA-test/result/star_ft/checkpoint_best.pth'
CUDA_VISIBLE_DEVICES=1 python3 evaluate.py \
--cfg-path lavis/projects/sevila/eval/star_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.star.vis_processor.eval.n_frms=16 \
run.batch_size_eval=4 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'