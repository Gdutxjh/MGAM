import torch
# import cfg # cfg是参数的预定于文件

# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath, map_location='cpu') 
#     model = checkpoint['model']  # 提取网络结构
#     model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
#     for parameter in model.parameters():
#         parameter.requires_grad = False
#     model.eval()
#     return model

# if __name__ == "__main__":
#     #利用trace把模型转化为pt
#     # trained_model = cfg.TRAINED_MODEL #cfg.TRAINED_MODEL表示resnet101.pth所在的位置
#     trained_model = "ckpt/resnet101.pth"
#     model = load_checkpoint(trained_model)
#     example = torch.rand(1, 3, 224, 224)
#     traced_script_module = torch.jit.trace(model, example)
#     traced_script_module.save('resnet101.pt')
#     output = traced_script_module(torch.ones(1, 3, 224, 224))
#     print(output)

# import torch
# import torchvision.models as models

# # 创建模型并保存整个模型
# # model = models.resnet101(pretrained=True)
# model = models.resnet101(pretrained="ckpt/resnet101.pth")
# # torch.save(model, 'ckpt/resnet101.pth')
# example = torch.rand(1, 3, 224, 224)
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save('ckpt/resnet101.pt')
# output = traced_script_module(torch.ones(1, 3, 224, 224))
# print(output)
n = 1000
a = []
for i in range(1000):
    a.append(str(i))
print(a)