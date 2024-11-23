import torch 
import random
import copy

def image_enhance(image):
    bs, t, l, w = image.shape
    m = int(l * 0.05)
    for k in range(bs):
        enhance_id = random.randint(1, 4)
        # enhance_id = 5
        # 添加随机噪声
        if enhance_id == 1:
            noise = torch.randn(t, l, w)*0.2
            image[k] += noise
        
        # 随机像素置零
        if enhance_id == 2:
            for i in range(l):
                idx = random.sample(range(0, l), m)
                image[k, :, i, idx] = 0
        
        # 随机整列置零
        if enhance_id == 3:
            idx = random.sample(range(0, l), m)
            image[k, :, :, idx] = 0
        
        # 随机整行置零
        if enhance_id == 4:
            idx = random.sample(range(0, l), m)
            image[k, :, idx, :] = 0
                
        # dkalklakf;a;fkllak; a;ha;asi;
        # if enhance_id == 5:
        #     idx = random.randint(0, l)
        #     image_1 = copy.deepcopy(image[k])
        #     image[k, :, :l-idx, :idx] = image_1[:, idx:, l-idx:]
        #     image[k, :, :l-idx, idx:] = image_1[:, idx:, :l-idx]
        #     image[k, :, l-idx:, :idx] = image_1[:, :idx, l-idx:]
        #     image[k, :, l-idx:, idx:] = image_1[:, :idx, :l-idx]
    return image

# if __name__ == "__main__":
#     # a = torch.ones((8, 3, 480, 480))
#     a = torch.rand((8, 3, 480, 480))
#     image_enhance(a)