import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import json
from decoder import CaptionDecoder
from utils.decoding_utils import greedy_decoding

# 加载预训练的 ResNet50 模型并移除最后的全连接层


def get_image_features(image_path, device):
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
    resnet = resnet.to(device)
    resnet.eval()

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = resnet(image_tensor)

    batch_size, channels, height, width = features.size()
    features = features.view(batch_size, channels, -1)
    features = features.mean(dim=2)

    return features

# 生成图像描述


def generate_caption(model, image_features, config, device, idx2word):
    model.eval()
    with torch.no_grad():
        sos_id = config["START_idx"]
        eos_id = config["END_idx"]
        pad_id = config["PAD_idx"]
        max_len = config["max_len"]

        image_features = image_features.to(device)

        caption_indices = greedy_decoding(
            model, image_features, sos_id, eos_id, pad_id, idx2word, max_len, device
        )

        captions = []
        for caption_indices_for_image in caption_indices:
            caption = ' '.join(caption_indices_for_image)
            captions.append(caption)

        return captions[0]  # 返回单个描述

# 加载词汇表


def load_vocab(word2idx_path):
    with open(word2idx_path, "r") as f:
        word2idx = json.load(f)
    idx2word = {int(idx): word for word, idx in word2idx.items()}
    return word2idx, idx2word

# 核心函数，接收图像并生成描述


def generate_caption_from_image(image, config_path="config.json", model_path="checkpoints/Oct-16_12-23-45/model_99.pth", vocab_path='dataset/word2idx.json', device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载配置文件
    with open(config_path, "r") as f:
        config = json.load(f)

    # 加载词汇表
    word2idx, idx2word = load_vocab(vocab_path)

    # 加载模型
    model = CaptionDecoder(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # 检查 image 是路径还是 PIL 图像，如果是路径则加载
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    image_path = "imgs/temp_image.jpg"
    image.save(image_path)

    image_features = get_image_features(image_path, device)

    # 生成并返回描述
    caption = generate_caption(model, image_features, config, device, idx2word)
    return caption
