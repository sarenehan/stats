#!/bin/bash
curl -O https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py --user
sudo python3 -m pip install numpy
sudo python3 -m pip install sklearn