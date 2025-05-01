# Setup script to run on a cloudlab node
sudo apt update
sudo apt install nvidia-cuda-toolkit
sudo apt install ubuntu-drivers-common
# ubuntu-drivers devices
sudo apt-get install alsa-utils

# ubuntu-drivers devices | grep "recommended" # get the recommended driver version
sudo apt install nvidia-driver-550 nvidia-utils-550 # change to the version that you got above, c240g5 wisconsin should be this though

# setup up conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init
source ~/.bashrc
conda init --all

wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_2/NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb
sudo apt update
sudo apt install ./NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb

# reboot the machine here then run below

# git clone https://github.com/KLi447/CS-598-FAL-Project
# cd CS-598-FAL-Project
# pip install .
#sudo reboot
