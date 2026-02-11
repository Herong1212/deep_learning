from torchvision import transforms, datasets
from model import NetWork
import torch

if __name__ == "__main__":
    # 图像预处理
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度值
            transforms.ToTensor(),  # 转换为张量
        ]
    )

    # 读取测试数据集
    test_dataset = datasets.ImageFolder(
        "/root/private/deep_learning/mnist_network/mnist_images/test",
        transform=transform,
    ) # ImageFolder
    print(f"训练数据集大小为：{len(test_dataset)}")

    # ? 此时为什么不需要加载 test_dataloader？？

    model = NetWork()  # 定义神经网络模型
    model.load_state_dict(
        torch.load("/root/private/deep_learning/mnist_network/mnist.pth")
    )  # 加载刚刚训练好的模型文件

    right = 0  # 保存正确识别的数量
    for i, (data, label) in enumerate(test_dataset):
        output = model(data)  # 将其中的数据data输入到模型
        predict = output.argmax(1).item()  # 选择概率最大标签的作为预测结果

        # 对比预测值predict和真实标签label
        if predict == label:
            right += 1
        else:
            # 将识别错误的样例打印了出来
            # image_path = test_dataset[i][0]
            # ! 注意，上面直接打印 test_dataset[i][0] 是错误的！打印的是 PIL.Image.Image 对象，而不是图片！应该为：
            image_path= test_dataset.samples[i][0]
            print(
                f"wrong case: predict = {predict} label = {label} img_path = {image_path}"
            )

    # 计算出测试效果
    # sample_num = len(test_dataset)
    # acc = right * 1.0 / sample_num
    # print("test accuracy = %d / %d = %.3lf" % (right, sample_num, acc))
    print(
        f"test accuracy = {right} / {len(test_dataset)} = {right / len(test_dataset)}"
    )
