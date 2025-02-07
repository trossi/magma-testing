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
