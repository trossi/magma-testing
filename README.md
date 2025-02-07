# Notes on testing MAGMA

## LUMI / MI250x (1 GCD)

### Installation

```bash
export EBU_USER_PREFIX=$PWD/EasyBuild
module load LUMI/24.03
module load partition/G
module load EasyBuild-user

eb magma-2.8.0-cpeGNU-24.03-rocm.eb -r
exit
```

### Testing

```bash
ml LUMI/24.03
ml partition/G
ml rocm/6.0.3

hipcc -std=c++14 --offload-arch=gfx90a -O3 -DHIP -lrocblas -lrocsolver eigh.cpp -o rocm6.0.3.x
sbatch --partition=dev-g --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-node=1 --time=01:00:00 -o 'rocm6.0.3.out' --wrap='./rocm6.0.3.x 3,100,200,400,800,1600'

hipcc -std=c++14 --offload-arch=gfx90a -O3 -DMAGMA -DHIP -lmagma eigh.cpp -o magma2.8.0_rocm6.0.3.x
sbatch --partition=dev-g --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-node=1 --time=01:00:00 -o 'magma2.8.0_rocm6.0.3.out' --wrap='./magma2.8.0_rocm6.0.3.x 3,100,200,400,800,1600,3200,6400,12800'
```

### Testing ROCm 6.2.2

```bash
ml LUMI/24.03
ml partition/G
ml rocm/6.2.2

hipcc -std=c++14 --offload-arch=gfx90a -O3 -DHIP -lrocblas -lrocsolver eigh.cpp -o rocm6.2.2.x
sbatch --partition=dev-g --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-node=1 --time=01:00:00 -o 'rocm6.0.3.out' --wrap='./rocm6.0.3.x 3,100,200,400,800,1600,3200'
```

## Mahti / A100

### Installation

```bash
ml cuda/11.5.0

git clone --branch v2.8.0 https://github.com/icl-utk-edu/magma.git
cd magma
grep -rl '^#!/usr/bin/env python$' . | xargs sed -i 's|^#!/usr/bin/env python$|#!/usr/bin/env python3|g'
cp make.inc-examples/make.inc.openblas make.inc

# Build on an interactive shell on a node
srun -p test --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --exclusive -t 1:00:00 --pty bash
export TMPDIR=/dev/shm
make -j128 lib/libmagma.so GPU_TARGET=Ampere OPENBLASDIR=$OPENBLAS_INSTALL_ROOT CUDADIR=$CUDA_INSTALL_ROOT
```

### Testing

```bash
ml cuda/11.5.0

nvcc -std=c++14 -arch=sm_80 -O3 -DCUDA -lcusolver eigh.cpp -o cuda11.5.0.x
sbatch -p gputest --nodes=1 --ntasks-per-node=1 --gres=gpu:a100:1 -t 0:15:00 -o cuda11.5.0.out --wrap='./cuda11.5.0.x 3,100,200,400,800,1600,3200,6400,12800'

nvcc -std=c++14 -arch=sm_80 -O3 -DMAGMA -DCUDA -lmagma -I$PWD/magma/include -L$PWD/magma/lib -Xcompiler \"-Wl,-rpath,$PWD/magma/lib\" eigh.cpp -o magma2.8.0_cuda11.5.0.x
sbatch -p gputest --nodes=1 --ntasks-per-node=1 --gres=gpu:a100:1 -t 0:15:00 -o magma2.8.0_cuda11.5.0.out --wrap='./magma2.8.0_cuda11.5.0.x 3,100,200,400,800,1600,3200,6400,12800'
```
