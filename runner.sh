JOB_HOME="$( cd "$( dirname "$0" )" && pwd )"

YYYYMMDDHH=$(date +%Y%m%d%H)

S3_BUCKET=flo-reco-dev
S3_OBJPATH=model/log-track2vec

S3_T2V_HOME="s3://$S3_BUCKET/$S3_OBJPATH"
S3_WORKING_HOME=$S3_T2V_HOME/$YYYYMMDDHH
S3_LATEST_HOME=$S3_T2V_HOME/latest

TRAIN_DATA_DIR=train.dat.files
META_DATA_DIR=meta.dat.files

S3_TRAIN_DATA_PATH=$S3_LATEST_HOME/$TRAIN_DATA_DIR
S3_META_DATA_PATH=$S3_LATEST_HOME/$META_DATA_DIR
S3_LATEST_OUTPUT=$S3_LATEST_HOME/output
S3_WORKING_LOG=$S3_WORKING_HOME/train_log
S3_WORKING_OUTPUT=$S3_WORKING_HOME/output


export LOCAL_DATA_HOME=$HOME/data
export LOCAL_TRAIN_DATA_PATH=$LOCAL_DATA_HOME/$TRAIN_DATA_DIR
export LOCAL_META_DATA_PATH=$LOCAL_DATA_HOME/$META_DATA_DIR

export LOCAL_TRAIN_DATA_FILE=$LOCAL_DATA_HOME/train.dat
export LOCAL_META_DATA_FILE=$LOCAL_DATA_HOME/meta.dat
export LOCAL_MODEL_OUTPUT=$JOB_HOME/output
export LOCAL_LOG_HOME=$JOB_HOME/log

if [ ! -d "$LOCAL_TRAIN_DATA_PATH" ]; then
    mkdir -p $LOCAL_TRAIN_DATA_PATH
fi

if [ ! -d "$LOCAL_META_DATA_PATH" ]; then
    mkdir -p $LOCAL_META_DATA_PATH
fi

if [ ! -d "$LOCAL_MODEL_OUTPUT" ]; then
    mkdir -p $LOCAL_MODEL_OUTPUT
fi

if [ ! -d "$LOCAL_LOG_HOME" ]; then
    mkdir -p $LOCAL_LOG_HOME
fi

function downdload {
    echo "===================================== download train and meta data ============================"
    # sync
    aws s3 sync $S3_TRAIN_DATA_PATH $LOCAL_TRAIN_DATA_PATH
    cat $LOCAL_TRAIN_DATA_PATH/*.json > $LOCAL_TRAIN_DATA_FILE
    rm -rf $LOCAL_TRAIN_DATA_PATH
    
    aws s3 sync $S3_META_DATA_PATH $LOCAL_META_DATA_PATH
    cat $LOCAL_META_DATA_PATH/*.json > $LOCAL_META_DATA_FILE
    rm -rf $LOCAL_META_DATA_PATH
}

function train {
    
    #aws s3 sync $S3_LATEST_OUTPUT $LOCAL_MODEL_OUTPUT
    
    WINDOW_SIZE=3
    NUM_CORES=$(nproc --all)
    NUM_THREAD=$((NUM_CORES * 1))
    PRINT_INTERVAL=5
    LOG_BUFFER_SIZE=100
    IR=0.1
    EPOCH=20
    NEG=10
    VERBOSE=2
    MEMORY=1
    DIM=200
    PRETRAINED_IR=0.5
    LOAD_PRETRAINED=1
    IR_UPDATE_RATE=100000
    lrUpdateRate

    echo "===================================== start track2vec ============================"
    echo ">> LOCAL_TRAIN_DATA_FILE: ${LOCAL_TRAIN_DATA_FILE}"
    echo ">> LOCAL_META_DATA_FILE: ${LOCAL_META_DATA_FILE}"
    echo ">> LOCAL_MODEL_OUTPUT: ${LOCAL_MODEL_OUTPUT}"
    echo ">> LOCAL_LOG_HOME: ${LOCAL_LOG_HOME}"
    echo ">> S3_LATEST_OUTPUT: ${S3_LATEST_OUTPUT}"
    echo ">> S3_WORKING_LOG: ${S3_WORKING_LOG}"
    echo ">> S3_WORKING_OUTPUT: ${S3_WORKING_OUTPUT}"
    echo ">> YYYYMMDDHH: ${YYYYMMDDHH}"
    echo ">> NUM_THREAD: ${NUM_THREAD}"
    echo ">> PRINT_INTERVAL: ${PRINT_INTERVAL}"
    echo ">> LOG_BUFFER_SIZE: ${LOG_BUFFER_SIZE}"
    echo ">> IR: ${WINDOW_SIZE}"
    echo ">> IR: ${WINDOW_SIZE}"
    echo ">> EPOCH: ${EPOCH}"
    echo ">> NEG: ${NEG}"
    echo ">> DIM: ${DIM}"
    echo ">> LOAD_PRETRAINED: ${LOAD_PRETRAINED}"
    echo ">> PRETRAINED_IR: ${PRETRAINED_IR}"
    echo ">> VERBOSE: ${VERBOSE}"
    echo ">> MEMORY: ${MEMORY}"
    
    
    nohup $JOB_HOME/track2vec train \
    -input $LOCAL_TRAIN_DATA_FILE \
    -meta $LOCAL_META_DATA_FILE \
    -ws $WINDOW_SIZE \
    -output $LOCAL_MODEL_OUTPUT \
    -locallog $LOCAL_LOG_HOME \
    -s3log $S3_WORKING_LOG \
    -yyyymmddhh $YYYYMMDDHH \
    -printInterval $PRINT_INTERVAL \
    -logBufferSize $LOG_BUFFER_SIZE \
    -thread $NUM_THREAD \
    -lr $IR \
    -epoch $EPOCH \
    -dim $DIM \
    -neg $NEG \
    -pretrained_lr $PRETRAINED_IR \
    -loadPretrained $LOAD_PRETRAINED \
    -memory $MEMORY \
    -lrUpdateRate $IR_UPDATE_RATE \
    -verbose $VERBOSE &
}

function nn {
    echo "===================================== similarity search ============================"

    DIM=200
    SIM_OUTPUT_HOME=$JOB_HOME/output
    
    GENRE_N_TREE=100
    GENRE_TREE_FILE=$SIM_OUTPUT_HOME/genre_tree.ann
    GENRE_IDX_FILE=$SIM_OUTPUT_HOME/genre_idx.ann
    GENRE_INPUT_VEC_FILE=$SIM_OUTPUT_HOME/genre_vec.json
    GENRE_NN_OUTPUT=$SIM_OUTPUT_HOME/genre_ann.json

    python $JOB_HOME/nn.py all \
    --id_name genre_id \
    --dim $DIM \
    --treeFile $GENRE_TREE_FILE \
    --idxFile $GENRE_IDX_FILE \
    --ntree $GENRE_N_TREE \
    --input $GENRE_INPUT_VEC_FILE \
    --output $GENRE_NN_OUTPUT

    ARTIST_N_TREE=1000
    ARTIST_TREE_FILE=$SIM_OUTPUT_HOME/artist_tree.ann
    ARTIST_IDX_FILE=$SIM_OUTPUT_HOME/artist_idx.ann
    ARTIST_INPUT_VEC_FILE=$SIM_OUTPUT_HOME/artist_vec.json
    ARTIST_NN_OUTPUT=$SIM_OUTPUT_HOME/artist_ann.json

    python $JOB_HOME/nn.py all \
    --id_name artist_id \
    --dim $DIM \
    --treeFile $ARTIST_TREE_FILE \
    --idxFile $ARTIST_IDX_FILE \
    --ntree $ARTIST_N_TREE \
    --input $ARTIST_INPUT_VEC_FILE \
    --output $ARTIST_NN_OUTPUT

    TRACK_N_TREE=10000
    TRACK_TREE_FILE=$SIM_OUTPUT_HOME/track_tree.ann
    TRACK_IDX_FILE=$SIM_OUTPUT_HOME/track_idx.ann
    TRACK_INPUT_VEC_FILE=$SIM_OUTPUT_HOME/track_vec.json
    TRACK_NN_OUTPUT=$SIM_OUTPUT_HOME/track_ann.json

    python $JOB_HOME/nn.py all \
    --id_name track_id \
    --dim $DIM \
    --treeFile $TRACK_TREE_FILE \
    --idxFile $TRACK_IDX_FILE \
    --ntree $TRACK_N_TREE \
    --input $TRACK_INPUT_VEC_FILE \
    --output $TRACK_NN_OUTPUT

    aws s3 sync $LOCAL_MODEL_OUTPUT $S3_WORKING_OUTPUT
}

TASK_NAME=$1

case ${TASK_NAME} in
    downdload)
        downdload
    ;;
    train)
        train
    ;;
    nn)
        nn
    ;;
    *)
        echo ">> Args ("$@") are not supported"
        exit 1
    ;;
esac