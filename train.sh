#CUDA_VISIBLE_DEVICES=0 python main.py --task toxin --model voxel --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task toxin --model seq --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task toxin --model mm --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task toxin --model mm --loss mlce

#CUDA_VISIBLE_DEVICES=0 python main.py --task anti --model voxel --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task anti --model seq --loss ce
CUDA_VISIBLE_DEVICES=0 python main.py --task all --model mm --loss mse
#CUDA_VISIBLE_DEVICES=0 python main.py --task anti --model mm --loss curri
#CUDA_VISIBLE_DEVICES=0 python main.py --task anti --model mm --loss logcosh
#CUDA_VISIBLE_DEVICES=0 python main.py --task anti --model mm --loss sl

#CUDA_VISIBLE_DEVICES=0 python main.py --task anti --model mmf --loss mlce

#CUDA_VISIBLE_DEVICES=0 python main.py --task mechanism --model voxel --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task mechanism --model seq --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task mechanism --model mm --loss ce
#CUDA_VISIBLE_DEVICES=0 python main.py --task mechanism --model mm --loss mlce

#CUDA_VISIBLE_DEVICES=0 python main.py --task mic --model mm --loss mse
