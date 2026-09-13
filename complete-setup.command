#!/bin/zsh
set -e

project_dir=${0:A:h}
cd "$project_dir"

print "DeepSeek V4.1 Flash：继续下载、校验并执行 1-token 测试。"
print "请保持此终端窗口打开。"
./complete-setup.sh

print "\n配置完成。可以关闭此窗口。"
read -k 1 '?按任意键关闭…'
