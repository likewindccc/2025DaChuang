#!/usr/bin/env python3
# 测试multiprocess库是否可用

print("="*60)
print("测试multiprocess库")
print("="*60)

try:
    import multiprocess as mp
    print(f"✓ multiprocess导入成功")
    print(f"  版本: {mp.__version__}")
    
    # 测试闭包序列化
    def outer():
        x = 10
        def inner(y):
            return x + y
        return inner
    
    func = outer()
    
    print("\n测试闭包序列化...")
    with mp.Pool(2) as pool:
        results = pool.map(func, [1, 2, 3])
    print(f"✓ 闭包序列化测试成功: {results}")
    print("\n结论: multiprocess可以正常工作!")
    
except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("\n需要安装: pip install multiprocess")
    
except Exception as e:
    print(f"✗ 运行错误: {e}")
    import traceback
    traceback.print_exc()

