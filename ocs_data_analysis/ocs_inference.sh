NB=2 # number of bidders
MB=30 # number of grid points for bid
MV=60 # number of grid points for value
MGR='(0.5,0.5,15)' # grid of mean parameter of gaussian in the units of discretization of bid
SGR='(0.5,0.5,15)' # grid of std parameter of gaussian in the units of discretization of bid
QTL=0.9 # quantile for confidence interval
BID_THR=20000 # cap on maximum bid to consider
SPLE_SIZE=125 # sub-sample size for sub-sampling estimation of confidence set
NB_SPLES=4 # how many sub-samples to draw
RESULT_DIR="results" # the directory to store results

OUT_F_PRE="${RESULT_DIR}/sharp_set_nb_${NB}_mb_${MB}_mv_${MV}_mgr_${MGR}_sgr_${SGR}_bid_thr_${BID_THR}"
TXT_RESULTS="${RESULT_DIR}/post_process_results_nb_${NB}_mb_${MB}_mv_${MV}_mgr_${MGR}_sgr_${SGR}_bid_thr_${BID_THR}.txt"
LOG_FILE_PRE="${RESULT_DIR}/log_file_nb_${NB}_mb_${MB}_mv_${MV}_mgr_${MGR}_sgr_${SGR}_bid_thr_${BID_THR}"

mkdir -p $RESULT_DIR

python parallel_common_value_ocs_sparse.py --nb $NB --mb $MB --mv $MV --mgr $MGR --sgr $SGR --out "${OUT_F_PRE}_001.npy" --bid_thr $BID_THR > "${LOG_FILE_PRE}_${1}.txt"

python combine_sweeps.py --mgr $MGR --sgr $SGR --out $OUT_F_PRE

python parallel_common_value_subsampling.py --nb $NB --mb $MB --mv $MV --mgr $MGR --sgr $SGR --out_pre "${OUT_F_PRE}_001" --bid_thr $BID_THR --sple_size $SPLE_SIZE  --nb_sples $NB_SPLES >> "${LOG_FILE_PRE}_${1}.txt"

python combine_sweeps_subsampling.py --mgr $MGR --sgr $SGR --out $OUT_F_PRE --nb_sples $NB_SPLES

python quantile_estimation_tolerance.py --qtl $QTL --in_tol "${OUT_F_PRE}.npy" --in_sub "${OUT_F_PRE}_subsampling.npy" --out "${OUT_F_PRE}_qtl.npy"

python ocs_postprocess.py --nb $NB --mb $MB --mv $MV --mgr $MGR --sgr $SGR --bid_thr $BID_THR --in "${OUT_F_PRE}.npy" --tol_file "${OUT_F_PRE}_qtl.npy" > $TXT_RESULTS
