#!/bin/bash

# 输入文件夹路径
input_dir="/home/luwen/Code/wzy/vimeo90k/BDx4/00001"
# 输出文件夹路径
output_dir="/home/luwen/Code/wzy/vimeo90k/BDx4_BD15/00001"

# 遍历输入文件夹中的所有子文件夹
for sub_dir in "$input_dir"/*; do
    # 检查是否是目录
    if [ -d "$sub_dir" ]; then
        # 获取子文件夹的名称
        sub_dir_name=$(basename "$sub_dir")
        # 创建输出目录
        mkdir -p "$output_dir/$sub_dir_name"

        # 进入子文件夹
        cd "$sub_dir" || exit

        # 执行FFmpeg命令
        /usr/bin/ffmpeg -framerate 10 -i im%1d.png -c:v libx264 -crf 0 lossless.mp4
        /usr/bin/ffmpeg -i lossless.mp4 -vcodec libx264 -crf 15 crf25.mp4
        /usr/bin/ffmpeg -ss 00:00:00 -t 00:00:10 -i crf25.mp4 -r 10 -start_number 0 "$output_dir/$sub_dir_name/im%1d.png"

        # 删除中间生成的MP4文件
        rm -f lossless.mp4 crf25.mp4
    fi
done

echo "All subfolders processed."


# # 遍历输入文件夹中的所有子文件夹（第一层）
# for sub_dir in "$input_dir"/*; do
#     # 检查是否是目录
#     if [ -d "$sub_dir" ]; then
#         # 遍历第一层目录中的所有子文件夹（第二层）
#         for sub_sub_dir in "$sub_dir"/*; do
#             # 检查是否是目录
#             if [ -d "$sub_sub_dir" ]; then
#                 # 获取第二层子文件夹的名称
#                 sub_sub_dir_name=$(basename "$sub_sub_dir")
#                 # 创建输出目录
#                 mkdir -p "$output_dir/$sub_sub_dir_name"

#                 # 遍历第二层目录中的所有子文件夹（第三层）
#                 for sub_sub_sub_dir in "$sub_sub_dir"/*; do
#                     # 检查是否是目录
#                     if [ -d "$sub_sub_sub_dir" ]; then
#                         # 获取第三层子文件夹的名称
#                         sub_sub_sub_dir_name=$(basename "$sub_sub_sub_dir")
#                         # 创建输出目录
#                         mkdir -p "$output_dir/$sub_sub_dir_name/$sub_sub_sub_dir_name"

#                         # 进入第三层子文件夹
#                         cd "$sub_sub_sub_dir" || exit

#                         # 执行FFmpeg命令
#                         /usr/bin/ffmpeg -framerate 10 -i im%1d.png -c:v libx264 -crf 0 lossless.mp4
#                         /usr/bin/ffmpeg -i lossless.mp4 -vcodec libx264 -crf 15 crf25.mp4
#                         /usr/bin/ffmpeg -ss 00:00:00 -t 00:00:10 -i crf25.mp4 -r 10 -start_number 0 "$output_dir/$sub_sub_dir_name/$sub_sub_sub_dir_name/im%1d.png"

#                         # 删除中间生成的MP4文件
#                         rm -f lossless.mp4 crf25.mp4
#                     fi
#                 done
#             fi
#         done
#     fi
# done
# echo "All subfolders processed."