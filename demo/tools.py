import subprocess
from pathlib import Path


def exe_python(files):
    # 定义命令行参数
    for image_name in files:
        # image_name = 'P1781.png'
        command = 'C:/Users/sheny/miniconda3/envs/openmmlab/python.exe D:/my_workspace/mmrotate/demo/image_demo.py D:/my_workspace/mmrotate/demo/images/' + image_name + ' D:/my_workspace/mmrotate/demo/configs/oriented_rcnn_r50_fpn_1x_dota_le90.py D:/my_workspace/mmrotate/demo/checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file D:/my_workspace/mmrotate/demo/results/' + image_name
    

        try:
            # 使用 subprocess.run 执行命令
            result = subprocess.run(command, capture_output=True, text=True)
        except e:
            print("报错信息： ", image_name, " ", command)


        # 输出命令执行的结果
        if result.returncode == 0:
            print("命令执行成功")
            print("标准输出:", result.stdout)
        else:
            print("命令执行失败")
            print("错误输出:", result.stderr)
        

if __name__ == '__main__':
    folder_path = Path('D:/my_workspace/mmrotate/demo/images')                      # 文件夹路径
    files = [file.name for file in folder_path.iterdir() if file.is_file()]         # 获取文件夹下所有文件（不包含文件夹）
    # print(files)      # 输出文件列表
    exe_python(files)