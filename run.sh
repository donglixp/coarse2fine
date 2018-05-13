DATANAME=$1
GPU_ID=$2

PWD_DIR=$(pwd)
WORK_DIR=$(dirname "$(readlink -f $0)")
cd $WORK_DIR
DATA_DIR=$WORK_DIR/data_model/$DATANAME

if [ $DATANAME = "geoqueries" ] ; then
    cd logical
    python preprocess.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -rnn_size 300 -word_vec_size 150 -decoder_input_size 150 -layers 1 -start_checkpoint_at 60 -learning_rate 0.005 -start_decay_at 0 -epochs 100 -global_attention "dot" -attn_hidden 0 -lock_dropout -dropout 0.5 -dropout_i 0.5
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -split test -model_path "$DATA_DIR/run.*/m_100*.pt"
fi

if [ $DATANAME = "atis" ] ; then
    cd logical
    python preprocess.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -rnn_size 250 -word_vec_size 200 -decoder_input_size 150 -layers 1 -start_checkpoint_at 60 -learning_rate 0.005 -start_decay_at 0 -epochs 100 -global_attention "dot" -attn_hidden 0 -lock_dropout -dropout 0.5 -dropout_i 0.5
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -split dev -model_path "$DATA_DIR/run.*/m_*.pt"
    MODEL_PATH=$(head -n1 $DATA_DIR/dev_best.txt)
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -split test -model_path "$MODEL_PATH"
fi

if [ $DATANAME = "django" ] ; then
    cd $DATANAME
    python preprocess.py -root_dir "$WORK_DIR/data_model/" -dataset django -src_words_min_frequency 3 -tgt_words_min_frequency 5
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -rnn_size 300 -word_vec_size 250 -decoder_input_size 200 -layers 1 -start_checkpoint_at 15 -learning_rate 0.002 -epochs 25 -global_attention "dot" -attn_hidden 0 -dropout 0.3 -dropout_i 0.3 -lock_dropout -copy_prb hidden
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -split dev -model_path "$DATA_DIR/run.*/m_*.pt"
    MODEL_PATH=$(head -n1 $DATA_DIR/dev_best.txt)
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -split test -model_path "$MODEL_PATH"
fi

if [ $DATANAME = "wikisql" ] ; then
    cd $DATANAME
    python preprocess.py -train_anno "$DATA_DIR/annotated_ent/train.jsonl" -valid_anno "$DATA_DIR/annotated_ent/dev.jsonl" -test_anno "$DATA_DIR/annotated_ent/test.jsonl" -save_data "$DATA_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -start_checkpoint_at 30 -split_type "incell" -epochs 40 -global_attention "general" -fix_word_vecs -dropout 0.5 -score_size 64 -attn_hidden 64 -rnn_size 250 -co_attention -data "$DATA_DIR" -save_dir "$DATA_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -split dev -data_path "$DATA_DIR" -model_path "$DATA_DIR/run.*/m_*.pt"
    MODEL_PATH=$(head -n1 $DATA_DIR/dev_best.txt)
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -split test -data_path "$DATA_DIR" -model_path "$MODEL_PATH"
fi

cd $PWD_DIR