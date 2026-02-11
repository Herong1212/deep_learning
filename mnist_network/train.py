import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from model import NetWork

# from torch import optim 如果 (torch.)optim.Adam() 前面加上 torch. 就可以不写这行了
# from torch import nn

if __name__ == "__main__":
    print("========== 1. 环境准备 ==========")
    # 检查是否有显卡，如果有，我们将数据移到 GPU，C++ 程序员通常很在意这个
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    # 图像预处理
    # 下面的 datasets.ImageFolder() 默认加载图片是 RGB 格式（3通道）。MNIST 是黑白的，所以用 Grayscale 强转为 1 通道。
    # ToTensor()：这是最关键的一步。
    #   1、输入：Python 的 PIL Image 对象 (整数, 0-255)。
    #   2、输出：PyTorch Tensor (浮点数, 0.0-1.0)。
    #   3、形状变化：(H, W, C) -> (C, H, W) (PyTorch 习惯把通道放在前面)。
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度值
            transforms.ToTensor(),  # 转换为张量
        ]
    )

    print("\n========== 2. 数据加载 ==========")
    # 读入并构造训练数据集
    train_dataset = datasets.ImageFolder("mnist_images/train", transform=transform)
    print(f"成功加载数据集！数据集总样本数 (len): {len(train_dataset)}")

    # 【调试】看看数据集里第0个数据长什么样
    sample_img, sample_label = train_dataset[0]
    print(f"单张样本形状 (Shape): {sample_img.shape} (C, H, W)")
    print(f"单张样本类型 (Type): {sample_img.dtype}")
    print(f"单张样本标签 (Label): {sample_label}")

    # 小批量的数据读入
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    print(f"DataLoader 批次数量: {len(train_dataloader)} (总数 / 64)")

    print("\n========== 3. 模型初始化 ==========")
    model = NetWork().to(device)  # 1.模型本身，即我们设计的神经网络，并移至 GPU
    print("模型结构：\n", model)
    # note optimizer：它持有模型所有参数的指针（引用）。调用 step() 时，它会遍历这些参数，执行 w = w - lr * grad。
    optimizer = torch.optim.Adam(model.parameters())  # 2.优化器，优化模型中的参数
    # note criterion：计算误差。注意 CrossEntropyLoss 内部自带了 Softmax，所以你的模型输出层不需要加 Softmax。
    criterion = torch.nn.CrossEntropyLoss()  # 3.损失函数，分类问题，使用交叉熵损失误差

    print("\n========== 4. 开始训练循环 (只打印第一个 Batch 的细节) ==========")
    # 进入模型的迭代循环
    for epoch in range(10):  # 外层循环，代表了整个训练数据集的遍历次数
        # 整个训练集要循环多少轮，是10次、20次或者100次都是可能的，

        # 内存循环使用train_loader，进行小批量的数据读取
        # enumerate 返回的是 (索引, 数据内容)
        for batch_idx, (data, label) in enumerate(train_dataloader):
            # 将数据移至设备 GPU
            data, label = data.to(device), label.to(device)

            # ---【核心调试区：深度观察数据变化】---
            if batch_idx == 0:  # 我们只在第一个批次打印，否则屏幕会炸
                print(f"\n[Debug] Batch {batch_idx} 数据流分析:")
                print(f"1. 输入数据 (Input) 形状: {data.shape}")
                print(
                    f"   解析: [Batch={data.shape[0]}, Channel={data.shape[1]}, Height={data.shape[2]}, Width={data.shape[3]}]"
                )
                print(f"2. 标签数据 (Label) 形状: {label.shape}")
                print(f"   解析: 这是一个包含 64 个整数的向量")

            # 内层每循环一次，就会进行一次梯度下降算法
            # 包括了5个步骤:
            # step1.计算神经网络的前向传播结果
            output = model(data)
            if batch_idx == 0:
                print(f"3. 模型输出 (Output) 形状: {output.shape}")
                print(f"   解析: [Batch=64, Class=10]。每个样本有 10 个得分。")
                print(
                    f"   示例: 第1个样本的得分向量: {output[0].detach().cpu().numpy()}"
                )
            # step2.计算output和标签label之间的损失loss
            loss = criterion(output, label)
            if batch_idx == 0:
                print(f"4. 损失值 (Loss): {loss.item()} (这是一个标量)")
                print(f"   Loss 的 grad_fn: {loss.grad_fn} (这是反向传播的链表头)")
            # step3.使用backward计算梯度
            loss.backward()
            # step4.使用optimizer.step更新参数
            optimizer.step()
            # step5.将梯度清零。
            # 原理：PyTorch 的梯度是累加 (Accumulate) 的。如果不清零，第二次循环的梯度 = 第一次的梯度 + 第二次的梯度，这会导致更新方向错误。
            optimizer.zero_grad()
            # 以上这5个步骤，是使用pytorch框架训练模型的定式，初学的时候，先记住就可以了

            # ---【调试结束】---

            # 每迭代100个小批量，就打印一次模型的损失，观察训练的过程
            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}/10 "
                    f"| Batch {batch_idx}/{len(train_dataloader)} "
                    f"| Loss: {loss.item():.4f}"
                )
    print("\n========== 5. 保存模型 ==========")
    torch.save(model.state_dict(), "mnist.pth")  # 保存模型
    print("模型参数已保存到 mnist.pth")
    print("本质上它是一个 OrderedDict (有序字典)，可以通过 torch.load 查看。")
