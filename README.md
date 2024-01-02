# さくらインターネット H100

## sshdの設定
- 設定を変更
  - `/etc/ssh/sshd_config`
    - `PubkeyAuthentication yes`
    - `PasswordAuthentication no`
  - `/etc/ssh/sshd_config.d/50-cloud-init.conf`
    - `PasswordAuthentication no`
- `sudo systemctl restart sshd`　で設定を反映

## ユーザ登録とパスワードの変更
```sh
sudo useradd -m newuser
sudo passwd newuser
sudo gpasswd -a newuser sudo
sudo su newuser
cd
mkdir .ssh
chmod 700 .ssh
touch .ssh/authorized_keys
chmod 600 .ssh/authorized_keys
cat somewhere >> .ssh/authorized_keys
```

## 各ユーザの初期設定
初期パスワードを変更＋デフォルトシェルをbashに変更
```sh
passwd
chsh -s /bin/bash
```

## nvidia-dirverのインストール

```sh
ubuntu-drivers devices
```
を実行し, ドライバーの検出をする.

```
== /sys/devices/pci0000:15/0000:15:01.0/0000:16:00.0/0000:17:00.0/0000:18:00.0 ==
modalias : pci:v000010DEd00002330sv000010DEsd000016C1bc03sc02i00
vendor   : NVIDIA Corporation
driver   : nvidia-driver-535-server - distro non-free
driver   : nvidia-driver-525-server - distro non-free
driver   : nvidia-driver-525 - distro non-free
driver   : nvidia-driver-525-open - distro non-free
driver   : nvidia-driver-535-server-open - distro non-free
driver   : nvidia-driver-535 - distro non-free recommended
driver   : nvidia-driver-535-open - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

ドライバーのインストール.
```bash
sudo apt-get update 
sudo apt install nvidia-driver-535 nvidia-dkms-535
```

## CUDA Toolkit 
2024年1月3日現在, torchのサポートが12.1までなので, 12.1を入れる.

https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=runfile_local
を使用した. 
指示に従う.
```
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

実行後,選択に迫られるので,

```
Continue
accept
```
の後, Driverのcheckは外して,

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 530.30.02                                                           │
│ + [X] CUDA Toolkit 12.1                                                      │
│   [X] CUDA Demo Suite 12.1                                                   │
│   [X] CUDA Documentation 12.1                                                │
│ - [ ] Kernel Objects                                                         │
│      [ ] nvidia-fs                                                           │
│   Options                                                                    │
│   Install                                                                    │
│
```

参考に,
```
cat /etc/os-release
```
で選ぶための必要な情報を確認する.

nvcc --version

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Feb__7_19:32:13_PST_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0
```
みたいな感じになるはず.

Driverのcheckをはずしそびれると,
```bash
nvidia-smi
```
実行時に,
```
Failed to initialize NVML: Driver/library version mismatch
```
となる.
```bash
cat /sys/module/nvidia/version
```
で確認すると,
```
530.30.02
```
となるので注意.



```bash
sudo apt install cuda-drivers-fabricmanager-535
```
すると pytorchが, cudaを認識してくれるようになる.

## python環境構築

```bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git
```

```bash
sudo apt install python3.10-venv
```

```bash
curl https://pyenv.run | bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv install 3.10
pyenv local 3.10.13
python -m venv venv
```

mpirunのため,
```bash
sudo apt-get install libopenmpi-dev
```

自分が使用したtorchのバージョン
```
torch==2.1.2+cu121
```

apexはいつも通り,
```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```

flash-attentionのインストールはいつも通り,
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

```
source venv/bin/activate
wandb login
```
wandbのloginをしておく.

pytorchがcudaを認識してくれるか確認をする.
```
import torch
print(torch.cuda.is_available())
```

必要なものをexpert
```
export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda-12.1/"
```


## マルチノード (2node)

マルチノードの起動時に必要. 
```bash 
sudo apt-get install openmpi-bin
```

mdx 同様, ssh -A でログインをする必要がある. 
ノード間sshできるか確認をすると良い. 

### マウント (他にいい方法がありそう?)

両方ノード
```bash 
sudo apt-get update
sudo apt-get install nfs-kernel-server
sudo apt-get install nfs-common
```

2つのノード, 156と157とする.

157で,
```bash 
sudo vim  /etc/exports
```
をして,

```
/mnt/taishi-work-space {156のipアドレス}(rw,sync,no_subtree_check)
```
を書き込み保存.

```
sudo exportfs -a
sudo systemctl restart nfs-kernel-server
```

156では,
```
sudo mount  153.126.239.157:/mnt/taishi-work-space /mnt/taishi-work-space
```
を実行. dirが存在しない場合は事前に作成をしておく.
```
sudo mkdir  /mnt/taishi-work-space
sudo chown taishi:taishi  /mnt/taishi-work-space
```

## nccl インストール

```bash
sudo apt install build-essential devscripts debhelper fakeroot
```

```bash
git clone https://github.com/NVIDIA/nccl.git
```

してきて, 

https://github.com/NVIDIA/nccl
のインストール方法に基本従う

https://github.com/NVIDIA/nccl/issues/1100

にあるように,`LD_LIBRARY_PATH` に加える必要がある.



## nccl-test

./build/all_reduce_perf -b 8 -e 256M -f 2 -g 8

```
# nThread 1 nGpus 8 minBytes 8 maxBytes 268435456 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 498607 on ubuntu-server device  0 [0x18] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid 498607 on ubuntu-server device  1 [0x2a] NVIDIA H100 80GB HBM3
#  Rank  2 Group  0 Pid 498607 on ubuntu-server device  2 [0x3a] NVIDIA H100 80GB HBM3
#  Rank  3 Group  0 Pid 498607 on ubuntu-server device  3 [0x5d] NVIDIA H100 80GB HBM3
#  Rank  4 Group  0 Pid 498607 on ubuntu-server device  4 [0x9a] NVIDIA H100 80GB HBM3
#  Rank  5 Group  0 Pid 498607 on ubuntu-server device  5 [0xab] NVIDIA H100 80GB HBM3
#  Rank  6 Group  0 Pid 498607 on ubuntu-server device  6 [0xba] NVIDIA H100 80GB HBM3
#  Rank  7 Group  0 Pid 498607 on ubuntu-server device  7 [0xdb] NVIDIA H100 80GB HBM3
```

## GLOO

1nodeの時はエラーが起きなかったが, 2nodeの時は,
`
[E ProcessGroupGloo.cpp:138] Gloo connectFullMesh failed with [../third_party/gloo/gloo/transport/tcp/pair.cc:144] no error
Gloo?
`
が起こる. 


GPU間通信は, NCCLが使われるが, 

https://github.com/NVIDIA/Megatron-LM/issues/435

にあるように, GLOOが必要, 

```bash
ifconfig
```

コマンド実行のために,
```bash
sudo apt install net-tools
```

GLOOで初期される前に, 

```python
os.environ['GLOO_SOCKET_IFNAME'] = 'bond0'
```
を差し込んでおく.
