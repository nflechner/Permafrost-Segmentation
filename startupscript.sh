#!/bin/bash

# Update the system packages
sudo apt-get update

apt-get install qt5-default
apt-get install tmux

# Install Python and pip
sudo apt-get install -y python3 python3-pip

# Install build dependencies
sudo apt-get install -y build-essential python3-dev

# Upgrade pip and setuptools
pip3 install --upgrade pip setuptools

# Install the wheel package
pip3 install wheel

# Create a virtual environment (optional but recommended)
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# enter my repo
git clone https://github.com/nflechner/Permafrost-Segmentation.git 
cd Permafrost-Segmentation 

# Install packages from requirements.txt
pip install -r requirements.txt
pip install rasterio


