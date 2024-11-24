#!/bin/bash
rocm_version=6.2.1
source /etc/profile.d/z00_lmod.sh
module swap PrgEnv-cray PrgEnv-gnu
module load rocm/$rocm_version
module load craype-accel-amd-gfx90a
module load cray-pmi/6.1.15
module load cray-libpals/1.2.11
module load libfabric/2.1

# Clear existing env vars
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

# Essential NCCL settings
export NCCL_DEBUG=WARN  # Changed from INFO to WARN to reduce output
#export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsi0
export NCCL_NET_GDR_LEVEL=5

# AMD specific settings
#export HSA_ENABLE_SDMA=0
export ROCR_VISIBLE_DEVICES=0,1,2,3
#export HSA_FORCE_FINE_GRAIN_PCIE=0

# Uncomment these 4 lines to go fast!
plugin_loc=/collab/usr/global/tools/rccl/$SYS_TYPE/rocm-6.2.1/install/lib
export LD_LIBRARY_PATH=${plugin_loc}:$LD_LIBRARY_PATH  # point to directory with plugin library
export NCCL_NET_GDR_LEVEL=3  # enable GPU Direct RDMA communication
export FI_CXI_ATS=0 # enable GPU Direct RDMA communication

# Only add these if needed for stability
#export FI_CXI_RX_MATCH_MODE=software  # For stability
export NCCL_ASYNC_ERROR_HANDLING=1    # For error recovery

#suggested by https://github.com/argonne-lcf/alcf-nccl-tests
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

#some potentially bad ideas
#export RCCL_KERNEL_FORCE_COPY=1  # Add this for better stability with ROCm
# Libfabric optimizations for large-scale
#export FI_CXI_RX_MATCH_MODE=software
#export FI_CXI_RDZV_PROTO=kafr    # Optimized rendezvous protocol
#export FI_CXI_MAX_NUM_KERNELS_IN_QUEUE=32
#export FI_CXI_MAX_OUTSTANDING_RX=16384
#export FI_CXI_DEFAULT_CQ_SIZE=131072  # Larger CQ size for more GPUs
#export NCCL_PROTO=simple        # Simpler protocol can be more stable
#export NCCL_ASYNC_ERROR_HANDLING=1  # Enable async error handling
# Torchrun specific settings
#export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Helps with debugging
#export TORCH_SHOW_CPP_STACKTRACES=1

# Libfabric specific settings
#export FI_CXI_DISABLE_CQ_HUGETLB=1    # Add this to avoid CQ issues
#export FI_LOG_LEVEL=warn
#export FI_CXI_OVFLOW_BUF_SIZE=8388608  # Increase overflow buffer size
#export NCCL_TREE_THRESHOLD=4294967296   # Increase tree algorithm threshold



# original launch script starts here
export TRAIN_DATA_PATH=/p/vast1/MLdata/project_gutenberg/processed
export VALID_DATA_PATH=/p/vast1/MLdata/project_gutenberg/processed/val
#export NCCL_SOCKET_IFNAME=hsi0
firsthost=$(flux getattr hostlist | /bin/hostlist -n 1)
#firsthost=$(hostname)
export MASTER_ADDRESS=$firsthost
#export MAIN_OPRT=25855
devices=4
nnodes=64
#nnodes=128
#export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export CUDA_VISIBLE_DEVICES='0,1,2,3'

echo $MASTER_ADDRESS
export NUMEXPR_MAX_THREADS=64

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
srun torchrun --nproc_per_node=$devices --nnodes=$nnodes --rdzv_backend c10d --rdzv_endpoint=$MASTER_ADDRESS \
--rdzv-id $RANDOM \
--local-ranks-filter 0 --role rank --tee 3 \
train.py --job.config_file train_configs/my_llama3_8b.toml
