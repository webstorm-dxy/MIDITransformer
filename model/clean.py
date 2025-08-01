import os
import glob
import shutil

def clean_pth_models():
    """
    清除所有已训练的.pth模型文件
    """
    # 查找所有.pth文件
    pth_files = glob.glob("./pth/*.pth")
    
    if not pth_files:
        print("未找到任何.pth模型文件")
        return
    
    print("找到以下.pth模型文件:")
    for file in pth_files:
        print(f"  - {file}")
    
    # 确认删除
    confirm = input("\n确定要删除所有这些.pth模型文件吗? (y/N): ")
    if confirm.lower() in ['y', 'yes']:
        for file in pth_files:
            try:
                os.remove(file)
                print(f"已删除: {file}")
            except Exception as e:
                print(f"删除 {file} 时出错: {e}")
        
        # # 如果目录为空，删除目录
        # try:
        #     os.rmdir("./pth")
        #     print("已删除空的pth目录")
        # except OSError:
        #     # 目录不为空或不存在，忽略
        #     pass
    else:
        print("取消删除操作")


def clean_tensorboard_logs():
    """
    清除TensorBoard日志文件
    """
    tensorboard_dir = "./runs"
    
    if not os.path.exists(tensorboard_dir):
        print("未找到TensorBoard日志目录")
        return
    
    print(f"找到TensorBoard日志目录: {tensorboard_dir}")
    
    # 确认删除
    confirm = input("\n确定要删除TensorBoard日志目录吗? (y/N): ")
    if confirm.lower() in ['y', 'yes']:
        try:
            shutil.rmtree(tensorboard_dir)
            print(f"已删除TensorBoard日志目录: {tensorboard_dir}")
        except Exception as e:
            print(f"删除TensorBoard日志目录时出错: {e}")
    else:
        print("取消删除操作")


def clean_all():
    """
    清除所有训练产物（模型文件和TensorBoard日志）
    """
    print("开始清理所有训练产物...")
    clean_pth_models()
    print()
    clean_tensorboard_logs()
    print("清理完成")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        option = sys.argv[1].lower()
        if option == "models":
            clean_pth_models()
        elif option == "logs":
            clean_tensorboard_logs()
        elif option == "all":
            clean_all()
        else:
            print("用法: python clean.py [models|logs|all]")
            print("  models - 仅清理模型文件")
            print("  logs   - 仅清理TensorBoard日志")
            print("  all    - 清理所有训练产物")
    else:
        print("用法: python clean.py [models|logs|all]")
        print("  models - 仅清理模型文件")
        print("  logs   - 仅清理TensorBoard日志")
        print("  all    - 清理所有训练产物")

