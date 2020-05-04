#!/bin/bash
#SBATCH -J climseg-cgpu
#SBATCH -C gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -A m1759
#SBATCH --exclusive
#SBATCH -t 58:00:00
#SBATCH -o %x-%j.out

# Job parameters
do_stage=false
ntrain=12
nvalid=0
ntest=0
batch=2
epochs=1
prec=32
grad_lag=1
scale_factor=0.1
loss_type=weighted #weighted_mean

# Parse command line options
while (( "$#" )); do
    case "$1" in
        --ntrain)
            ntrain=$2
            shift 2
            ;;
        --nvalid)
            nvalid=$2
            shift 2
            ;;
        --ntest)
            ntest=$2
            shift 2
            ;;
        --epochs)
            epochs=$2
            shift 2
            ;;
        --dummy)
            other_train_opts="--dummy_data"
            shift
            ;;
        -*|--*=)
            echo "Error: Unsupported flag $1" >&2
            exit 1
            ;;
    esac
done

#load modules
module load tensorflow/gpu-1.15.0-py37

export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export HDF5_USE_FILE_LOCKING=FALSE

# Setup directories
datadir=/global/cscratch1/sd/sfarrell/climate-seg-benchmark/data/climseg-data-duplicated
ls ${datadir}
cp -r ${datadir} /tmp
datadir=/tmp/climseg-data-duplicated
ls ${datadir}
scratchdir=${datadir} # no staging
run_dir=/tmp
out_dir=$SCRATCH/climate-seg-benchmark/run_cgpu/

# Prepare the run directory
cp stage_in_parallel.sh ${run_dir}/
cp ../utils/parallel_stagein.py ${run_dir}/
cp ../utils/graph_flops.py ${run_dir}/
cp ../utils/tracehook.py ${run_dir}/
cp ../utils/common_helpers.py ${run_dir}/
cp ../utils/data_helpers.py ${run_dir}/
cp ../deeplab-tf/deeplab-tf-train.py ${run_dir}/
cp ../deeplab-tf/deeplab-tf-inference.py ${run_dir}/
cp ../deeplab-tf/deeplab_model.py ${run_dir}/
cd ${run_dir}

metrics="\
sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,\
smsp__cycles_elapsed.sum,\
smsp__cycles_elapsed.sum.per_second,\
lts__t_sectors_aperture_sysmem_op_read.sum,\
lts__t_sectors_aperture_sysmem_op_write.sum,\
smsp__pipe_tensor_op_hmma_cycles_active.sum,\
smsp__pipe_tensor_op_hmma_cycles_active.sum.per_second,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_atom.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_red.sum,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
lts__t_sectors_op_atom.sum,\
lts__t_sectors_op_red.sum,\
dram__sectors_read.sum,\
dram__sectors_write.sum\
"

profilestring="/project/projectdirs/m1759/nsight-compute-2019.5.0.15/nv-nsight-cu-cli --profile-from-start off --metrics ${metrics} -f -o full_profile"

# Stage data if relevant
if [ "${scratchdir}" != "${datadir}" ]; then
    if $do_stage; then
        cmd="srun --mpi=pmi2 -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 80 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${ntrain} ${nvalid} ${ntest}"
        echo ${cmd}
        ${cmd}
    fi
else
    echo "Scratchdir and datadir is the same, no staging needed!"
fi

# Run the training
if [ $ntrain -ne 0 ]; then
    echo "Starting Training"
    srun -u --cpu_bind=cores ${profilestring} python -u deeplab-tf-train.py \
        --datadir_train ${scratchdir}/train \
        --train_size ${ntrain} \
        --datadir_validation ${scratchdir}/validation \
        --validation_size ${nvalid} \
        --chkpt_dir checkpoint.fp${prec}.lag${grad_lag} \
        --disable_checkpoint \
        --epochs $epochs \
        --fs "global" \
        --loss $loss_type \
        --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=${grad_lag} \
        --model "resnet_v2_50" \
        --scale_factor $scale_factor \
        --batch $batch \
        --decoder "deconv1x" \
        --device "/device:cpu:0" \
        --dtype "float${prec}" \
        --label_id 0 \
        --data_format "channels_first" \
        $other_train_opts |& tee out.fp${prec}.lag${grad_lag}.train
fi

rm -rf ${datadir}
cp -r ${run_dir} ${out_dir}

if [ $ntest -ne 0 ]; then
    echo "Starting Testing"
    srun -u --cpu_bind=cores ${profilestring} flops_inference python -u deeplab-tf-inference.py \
        --datadir_test ${scratchdir}/test \
        --chkpt_dir checkpoint.fp${prec}.lag${grad_lag} \
        --test_size ${ntest} \
        --output_graph deepcam_inference.pb \
        --output output_test_5 \
        --fs "local" \
        --loss $loss_type \
        --model "resnet_v2_50" \
        --scale_factor $scale_factor \
        --batch $batch \
        --decoder "deconv1x" \
        --device "/device:cpu:0" \
        --dtype "float${prec}" \
        --label_id 0 \
        --data_format "channels_last" |& tee out.fp${prec}.lag${grad_lag}.test
fi
