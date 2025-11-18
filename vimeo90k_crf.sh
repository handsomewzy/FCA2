# 输入文件夹的基础路径
base_input_dir="/home/luwen/Code/wzy/vimeo90k/BDx4"
# 输出文件夹的基础路径
base_output_dir="/home/luwen/Code/wzy/vimeo90k/BDx4_BD35"
# 要执行的脚本路径
script_to_run="/home/luwen/Code/wzy/CAD2VSR/data_crf.sh" # 替换为你实际的sh文件路径

# 遍历 00001 到 00096 的文件夹
for i in $(seq -w 8 9); do
    # 构建输入和输出文件夹路径
    input_dir="$base_input_dir/$(printf '%05d' $i)"
    output_dir="$base_output_dir/$(printf '%05d' $i)"

    # 检查输入目录是否存在
    if [ -d "$input_dir" ]; then
        # 创建输出目录
        mkdir -p "$output_dir"

        # 进入输入目录
        cd "$input_dir" || exit

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
                /usr/bin/ffmpeg -i lossless.mp4 -vcodec libx264 -crf 35 crf25.mp4
                /usr/bin/ffmpeg -ss 00:00:00 -t 00:00:10 -i crf25.mp4 -r 10 -start_number 0 "$output_dir/$sub_dir_name/im%1d.png"

                # 删除中间生成的MP4文件
                rm -f lossless.mp4 crf25.mp4
            fi
        done

        echo "All subfolders processed."

        # 执行指定的 .sh 脚本
        # bash "$script_to_run" "$input_dir" "$output_dir"
    else
        echo "目录不存在: $input_dir"
    fi
done