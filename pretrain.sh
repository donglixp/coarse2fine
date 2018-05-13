DATANAME=$1
GPU_ID=$2

PWD_DIR=$(pwd)
WORK_DIR=$(dirname "$(readlink -f $0)")
cd $WORK_DIR
DATA_DIR=$WORK_DIR/data_model/$DATANAME
MODEL_PATH=$DATA_DIR/pretrain.pt

if [ $DATANAME = "geoqueries" ] ; then
    cd logical
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -split test -model_path "$MODEL_PATH"
fi

if [ $DATANAME = "atis" ] ; then
    cd logical
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -split test -model_path "$MODEL_PATH"
fi

if [ $DATANAME = "django" ] ; then
    cd $DATANAME
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -root_dir "$WORK_DIR/data_model/" -dataset $DATANAME -split test -model_path "$MODEL_PATH"
fi

if [ $DATANAME = "wikisql" ] ; then
    cd $DATANAME
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py -split test -data_path "$DATA_DIR" -model_path "$MODEL_PATH"
fi

cd $PWD_DIR