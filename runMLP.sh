modelseed_list=(402025 102025 202025) # 302025 3070 4080 5090 20250101 72 46852)
N_list=(17) # 21 33 65 129)
h_list=(0.25) # 0.20 0.125 0.0625 0.03125)

for modelseed in "${modelseed_list[@]}"; do
    for i in "${!N_list[@]}"; do
        N=${N_list[$i]}
        h=${h_list[$i]}

        start_time=$(date +%s)

        CUDA_VISIBLE_DEVICES=0 python train.py --model MLP --data mnist --epochs 10 --h $h --N $N --M 1000 --modelseed $modelseed &
        CUDA_VISIBLE_DEVICES=1 python train.py --model MLP --data ct --epochs 30 --h $h --N $N --M 3000 --modelseed $modelseed &
        CUDA_VISIBLE_DEVICES=2 python train.py --model MLP --data workloads --epochs 30 --h $h --N $N --M 3000 --modelseed $modelseed &
        
        wait

        CUDA_VISIBLE_DEVICES=0 python train.py --model MLP --data cifar10 --epochs 20 --h $h --N $N --M 3000 --modelseed $modelseed &
        CUDA_VISIBLE_DEVICES=1 python train.py --model MLP --data adult --epochs 15 --h $h --N $N --M 3000 --modelseed $modelseed &
        CUDA_VISIBLE_DEVICES=2 python train.py --model MLP --data protein --epochs 30 --h $h --N $N --M 3000 --modelseed $modelseed &

        wait

        CUDA_VISIBLE_DEVICES=0 python train.py --model MLP --data msd --epochs 10 --h $h --N $N --M 3000 --modelseed $modelseed

        wait

        end_time=$(date +%s)
        elapsed_time=$((end_time - start_time))
        echo "Modelseed $modelseed N $N completed in $elapsed_time seconds." >> runMLP.log
    done
done

# for modelseed in "${modelseed_list[@]}"; do
#     start_time=$(date +%s)

#     CUDA_VISIBLE_DEVICES=0 python train.py --model MLP --data sin --epochs 45 --h 0.005 --N 401 --M 1000 --modelseed $modelseed &
#     CUDA_VISIBLE_DEVICES=1 python train.py --model MLP --data tru --epochs 45 --h 0.005 --N 401 --M 1000 --modelseed $modelseed &
#     CUDA_VISIBLE_DEVICES=2 python train.py --model MLP --data zoi --epochs 45 --h 0.005 --N 401 --M 1000 --modelseed $modelseed
#     wait

#     end_time=$(date +%s)
#     elapsed_time=$((end_time - start_time))
#     echo "Modelseed $modelseed completed in $elapsed_time seconds." >> runMLP.log
# done

# CUDA_VISIBLE_DEVICES=0 python train.py --model MLP --data sin --epochs 25 --h 0.02 --N 401 --M 1000 --modelseed 402025 &
# CUDA_VISIBLE_DEVICES=1 python train.py --model MLP --data tru --epochs 12 --h 0.01 --N 401 --M 1000 --modelseed 402025 &
# CUDA_VISIBLE_DEVICES=2 python train.py --model MLP --data zoi --epochs 45 --h 0.02 --N 401 --M 1000 --modelseed 102025