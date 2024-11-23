import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from data.PathVQA_dataset import PathVQA_datset, PathVQA_CPT_datset, VQA_RAD_datset, ROCO_dataset, VQA_SLAKE_datset
from data.PathVQA_dataset import PathVQA_pretrain, VQA_RAD_pretrain
from data.PathVQA_dataset import VQA_RAD_SWIN
from data.EVQA_dataset import evqa_dataset
from data.transform.randaugment import RandomAugment
from data.retrieval_dataset import SLAKE_datset

def create_dataset(config, keywords=None, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ]) 

    # if config['swin_config']:
    #     if config == 'VQA_RAD':
    #         train_dataset = VQA_RAD_datset(config=config, transform=transform_test, split="train")
    #         test_dataset = VQA_RAD_datset(config=config, transform=transform_test, split="test")
    #         return train_dataset, test_dataset, test_dataset
    
    if config['dataset'] == 'PathVQA':
        if config["pretrain"]:
            train_dataset = PathVQA_pretrain(config=config, transform=transform_train, split="train")
            val_dataset = PathVQA_pretrain(config=config, transform=transform_test, split="test")
            test_dataset = PathVQA_pretrain(config=config, transform=transform_test, split="test")
            return train_dataset, val_dataset, test_dataset
        
        train_dataset = PathVQA_datset(config=config, transform=transform_train, split="train")
        val_dataset = PathVQA_datset(config=config, transform=transform_test, split="test")
        test_dataset = PathVQA_datset(config=config, transform=transform_test, split="test")
        return train_dataset, val_dataset, test_dataset
    
    if config['dataset'] == 'VQA_RAD':
        if config['pretrain']:
            train_dataset = VQA_RAD_pretrain(config=config, transform=transform_test, split="train")
            test_dataset = VQA_RAD_pretrain(config=config, transform=transform_test, split="test")
            return train_dataset, test_dataset, test_dataset

        train_dataset = VQA_RAD_datset(config=config, transform=transform_test, split="train")
        test_dataset = VQA_RAD_datset(config=config, transform=transform_test, split="test")
        return train_dataset, test_dataset, test_dataset
    
    if config['dataset'] == 'VQA_SLAKE':
        train_dataset = VQA_SLAKE_datset(config=config, transform=transform_test, split="train")
        val_dataset = VQA_SLAKE_datset(config=config, transform=transform_test, split="val")
        test_dataset = VQA_SLAKE_datset(config=config, transform=transform_test, split="test")
        return train_dataset, test_dataset, test_dataset
    
    if config['dataset'] == 'ROCO':
        train_dataset = ROCO_dataset(split="train", tfm=transform_train, args=config, mode = 'train')
        val_dataset = ROCO_dataset(split="val", tfm=transform_train, args=config, mode = 'val')
        test_dataset = ROCO_dataset(split="test", tfm=transform_train, args=config, mode = 'test')
        return train_dataset, val_dataset, test_dataset
    
    if config["dataset"] == "EVQA":
        train_dataset = evqa_dataset(transform_train, config, split="train")
        val_dataset = evqa_dataset(transform_train, config, split="val")
        test_dataset = evqa_dataset(transform_train, config, split="test")
        return train_dataset, val_dataset, test_dataset
    
    if config['dataset'] == 'SLAKE':
        train_dataset = SLAKE_datset(config=config, transform=transform_test, split="train")
        val_dataset = SLAKE_datset(config=config, transform=transform_test, split="val")
        test_dataset = SLAKE_datset(config=config, transform=transform_test, split="test")
        return train_dataset, test_dataset, test_dataset
    # if config['dataset'] == 'ROCO_pretrain':
    #     train_dataset = ROCO_pretain_dataset(split="train", tfm=transform_train, args=config, keywords=keywords, mode = 'train')
    #     val_dataset = ROCO_pretain_dataset(split="val", tfm=transform_train, args=config, keywords=keywords, mode = 'val')
    #     test_dataset = ROCO_pretain_dataset(split="test", tfm=transform_train, args=config, keywords=keywords, mode = 'test')
    #     return train_dataset, val_dataset, test_dataset
    
def create_CPT_dataset(config, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        

    if config['dataset'] == 'PathVQA':
        CPT_dataset = PathVQA_CPT_datset(config=config, transform=transform_train, split="train")
        return CPT_dataset

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train  in zip(datasets,samplers,batch_size,num_workers,is_trains):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
