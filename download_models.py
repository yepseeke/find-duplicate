import torchvision.models as models


def download_all_weights():
    print("Downloading weights for:")

    models_dict = {
        "resnet152": models.resnet152(weights=models.ResNet152_Weights.DEFAULT),
        "vgg19_bn": models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT),
        "densenet201": models.densenet201(weights=models.DenseNet201_Weights.DEFAULT),
        "inception_v3": models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=False),
        "efficientnet_b7": models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT),
    }

    for name in models_dict:
        print(f"✔️  {name}")


if __name__ == "__main__":
    download_all_weights()
