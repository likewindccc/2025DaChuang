#!/usr/bin/env python3
# 测试multiprocess库是否能正常工作

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
    with mp.Pool(2) as pool:
        results = pool.map(func, [1, 2, 3])
    print(f"✓ 闭包序列化测试成功: {results}")
    print(f"✓ multiprocess可以正常使用！")
    
except Exception as e:
    print(f"✗ 错误: {type(e).__name__}: {e}")
    print("\n如果提示ModuleNotFoundError，请运行：")
    print("  pip install multiprocess==0.70.17")

