import re
import os

def extract_and_sort_m(file_path):
    m_values = []
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    pattern = re.compile(r'Tensor\(paddle\.Size\(\[(\d+),')
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # 查找每行第一个匹配项
                match = pattern.search(line)
                if match:
                    # 提取第一个捕获组（即M的值）
                    m_val = int(match.group(1))
                    m_values.append(m_val)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # 去重
    unique_m_values = list(set(m_values))
    
    # 排序
    unique_m_values.sort()
    
    # 输出结果
    print("Sorted unique M values:")
    print(unique_m_values)
    print(f"\nTotal unique count: {len(unique_m_values)}")

if __name__ == "__main__":
    # 假设脚本在 test/custom_api/ 目录下运行，或者直接指定相对路径
    # 根据用户当前上下文，文件路径相对于 workspace root
    target_file = 'test/custom_api/moe_permute.txt'
    extract_and_sort_m(target_file)