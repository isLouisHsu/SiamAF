#! bin/bash
### 
# @Description: 
 # @Version: 1.0.0
 # @Author: louishsu
 # @E-mail: is.louishsu@foxmail.com
 # @Date: 2019-12-13 11:58:11
 # @LastEditTime: 2019-12-16 19:33:18
 # @Update: 
 ###
cd ../main
rm -rf ../logs/siamrpn/*
python siamrpn.py -m train