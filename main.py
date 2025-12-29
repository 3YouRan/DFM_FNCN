import os
import sys
import config as cfg
from train import run_training
from validation import run_validation
from train_decoder import run_decoder_training
from visualize_rule import run_visualization


def main():
    print("=" * 60)
    print("DFM-FNCN 全流程自动化脚本")
    print("=" * 60)

    # 1. 训练模型 (Step 1)
    # run_training 会返回新创建的运行目录路径
    run_dir = run_training()

    

    if not run_dir or not os.path.exists(run_dir):
        print("训练失败或未返回有效的运行目录。退出。")
        sys.exit(1)

    print(f"\n[Main] 训练完成。运行目录: {run_dir}")

    # 2. 评估模型 (Step 2)
    run_validation(run_dir)

    # 3. 如果是 DFM_FNCN 模型，继续执行可解释性步骤
    if cfg.MODEL_TYPE == 'DFM_FNCN':
        print("\n[Main] 检测到 DFM_FNCN 模型，开始可解释性流程...")

        # 3.1 训练解码器 (Step 3)
        run_decoder_training(run_dir)

        # 3.2 可视化规则 (Step 4)
        run_visualization(run_dir)
    else:
        print(f"\n[Main] 模型类型为 {cfg.MODEL_TYPE}，跳过解码器训练和规则可视化。")

    print("\n" + "=" * 60)
    print(f"全流程结束！所有结果已保存在: {run_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()