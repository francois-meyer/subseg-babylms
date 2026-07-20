# Subword Segmental Language Models

Implementation of SubSegGPT and SubSegDeBERTa, developed for the 2026 BabyLM Challenge.

---

## 1. Installation

```bash
pip install -r requirements.txt
```

---

## 2. Training

### SubSegGPT

```bash
python -u train_subseggpt.py \
    --track strict \
    --output_dir outputs/subseggpt \
    --hidden_size 768 --num_hidden_layers 12 --num_attention_heads 12 --intermediate_size 3072 \
    --char_decoder_hidden_size 256 --char_decoder_num_layers 1 --char_embed_dim 128 \
    --max_seg_len 5 --lex_vocab_size 10000 --lex_min_count 5 \
    --block_size 1024 \
    --batch_size 16 --gradient_accumulation_steps 1 \
    --learning_rate 5e-4 --weight_decay 0.01 --warmup_ratio 0.1 --lr_scheduler_type cosine \
    --epochs 10 --dropout 0.1 \
    --save_strategy epoch --logging_steps 100 --dataloader_num_workers 4 --seed 42
```

### SubSegDeBERTa

```bash
python -u train_subsegdeberta.py \
    --track strict \
    --output_dir outputs/subsegdeberta \
    --hidden_size 768 --num_hidden_layers 12 --num_attention_heads 12 --intermediate_size 3072 \
    --word_context_hidden_size 768 --char_decoder_hidden_size 256 --char_embed_dim 128 \
    --max_position_embeddings 1024 \
    --max_seg_len 5 --lex_vocab_size 10000 --lex_min_count 5 \
    --mlm_probability 0.3 --loss_normalization word --max_word_len 20 \
    --block_size 1022 \
    --batch_size 16 --gradient_accumulation_steps 1 \
    --learning_rate 5e-4 --weight_decay 0.01 --warmup_ratio 0.1 --lr_scheduler_type cosine \
    --epochs 10 --dropout 0.1 \
    --save_strategy epoch --logging_steps 100 --dataloader_num_workers 4 --seed 42
```

## 3. Evaluation

Evaluation runs on top of the BabyLM 2026 evaluation pipeline, which you have to clone and
download evaluation data for:

```bash
git clone https://github.com/babylm-org/babylm-eval.git
cd babylm-eval/strict
pip install -r requirements.txt
python -m scripts.download_evals    
cd ../..
```

> **Patch the official pipeline to work with our models.** 
> To evaluate SubSegGPT on zero-shot tasks, change one line in the official pipeline:
> Change `babylm-eval/strict/evaluation_pipeline/sentence_zero_shot/dataset.py` from
> ```
> phrase_mask = [0 for _ in range(len(tokens))]   # before
> ```
> to
> ```
> phrase_mask = [1 for _ in range(len(tokens))]   # after
> ```


Then set up the environment and run the evaluations from inside the pipeline directory:

```bash
export PYTHONPATH="$PWD:$PWD/babylm-eval/strict:$PYTHONPATH"
cd babylm-eval/strict

MODEL=../../outputs/subseggpt # trained model directory
RUN=../..                             
DATA=evaluation_data/full_eval
RESULTS=results
```

### SubSegGPT

```bash
# Zero-shot minimal pairs
for TASK_DIR in "blimp blimp_filtered" "blimp supplement_filtered" "ewok ewok_filtered" \
                "entity_tracking entity_tracking" "comps comps"; do
  set -- $TASK_DIR
  python -u $RUN/run_eval_subseggpt.py zero_shot \
      --model_path_or_name $MODEL --backend causal \
      --task $1 --data_path $DATA/$2 \
      --batch_size 32 --save_predictions --revision_name main
done

# GlobalPIQA
for SPLIT in global_piqa_parallel global_piqa_nonparallel; do
  python -u $RUN/run_eval_subseggpt.py global_piqa \
      --model_path_or_name $MODEL --backend causal \
      --task $SPLIT --data_path $DATA/$SPLIT \
      --batch_size 32 --save_predictions --revision_name main
done

# Reading times
python -u $RUN/run_eval_subseggpt.py reading \
    --model_path_or_name $MODEL --backend causal \
    --data_path $DATA/reading/reading_data.csv \
    --output_dir $RESULTS --revision_name main

# Age of Acquisition
python -u $RUN/run_eval_subseggpt.py aoa \
    --model_name $MODEL --backend causal \
    --word_path $DATA/aoa/cdi_childes.json \
    --track_name non-strict-small \
    --output_dir $RESULTS

# (Super)GLUE finetuning
python -u $RUN/run_eval_subseggpt.py finetune \
    --model_name_or_path $MODEL \
    --task boolq --num_labels 2 --batch_size 16 --gradient_accumulation 2 \
    --num_epochs 10 --sequence_length 1024 \
    --metrics accuracy f1 mcc --metric_for_valid accuracy \
    --results_dir $RESULTS --seed 42 --revision_name main
```

### SubSegDeBERTa

```bash
MODEL=../../outputs/subsegdeberta

# Zero-shot, including GlobalPIQA
for TASK_DIR in "blimp blimp_filtered" "blimp supplement_filtered" "ewok ewok_filtered" \
                "entity_tracking entity_tracking" "comps comps" \
                "global_piqa_parallel global_piqa_parallel" \
                "global_piqa_nonparallel global_piqa_nonparallel"; do
  set -- $TASK_DIR
  python -u $RUN/run_eval_subsegdeberta.py zero_shot \
      --model_path_or_name $MODEL --backend mlm \
      --task $1 --data_path $DATA/$2 \
      --non_causal_batch_size 32 --save_predictions --revision_name main
done

# Reading times
python -u $RUN/run_eval_subsegdeberta.py reading \
    --model_path_or_name $MODEL --backend mlm \
    --data_path $DATA/reading/reading_data.csv \
    --number_of_mask_tokens_to_append 3 \
    --output_dir $RESULTS --revision_name main

# Age of Acquisition
python -u $RUN/run_eval_subsegdeberta.py aoa \
    --model_name $MODEL --backend mlm \
    --word_path $DATA/aoa/cdi_childes.json \
    --track_name non-strict-small \
    --output_dir $RESULTS
```


---
