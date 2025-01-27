modelseed_list=(402025 102025 202025 302025 3070 4080 5090 20250101 72 46852)
actfunc_list=('relu' 'cos' 'tanh' 'sigmoid')

for modelseed in "${modelseed_list[@]}"; do
    for actfunc in "${actfunc_list[@]}"; do
        start_time=$(date +%s)

        CUDA_VISIBLE_DEVICES=0 python train.py --model RFMLP --data mnist --epochs 10 --M 1000 --modelseed $modelseed --actfunc $actfunc &
        CUDA_VISIBLE_DEVICES=1 python train.py --model RFMLP --data cifar10 --epochs 20 --M 3000 --modelseed $modelseed --actfunc $actfunc &
        CUDA_VISIBLE_DEVICES=2 python train.py --model RFMLP --data adult --epochs 15 --M 3000 --modelseed $modelseed --actfunc $actfunc &
        CUDA_VISIBLE_DEVICES=3 python train.py --model RFMLP --data protein --epochs 30 --M 3000 --modelseed $modelseed --actfunc $actfunc &
        CUDA_VISIBLE_DEVICES=4 python train.py --model RFMLP --data ct --epochs 30 --M 3000 --modelseed $modelseed --actfunc $actfunc &
        CUDA_VISIBLE_DEVICES=5 python train.py --model RFMLP --data workloads --epochs 30 --M 3000 --modelseed $modelseed --actfunc $actfunc &
        CUDA_VISIBLE_DEVICES=6 python train.py --model RFMLP --data msd --epochs 10 --M 3000 --modelseed $modelseed --actfunc $actfunc
        
        wait

        CUDA_VISIBLE_DEVICES=0 python train.py --model RFMLP --data sin --epochs 45 --M 1000 --actfunc $actfunc --modelseed $modelseed &
        CUDA_VISIBLE_DEVICES=1 python train.py --model RFMLP --data tru --epochs 45 --M 1000 --actfunc $actfunc --modelseed $modelseed &
        CUDA_VISIBLE_DEVICES=2 python train.py --model RFMLP --data zoi --epochs 45 --M 1000 --actfunc $actfunc --modelseed $modelseed

        wait

        end_time=$(date +%s)
        elapsed_time=$((end_time - start_time))
        echo "Modelseed $modelseed with activation function $actfunc completed in $elapsed_time seconds." >> runNmlp.log
    done
done