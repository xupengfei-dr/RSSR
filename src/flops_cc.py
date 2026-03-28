from fvcore.nn import FlopCountAnalysis
from fvcore.nn import parameter_count

model.eval()
inputs = {
    "pixel_values": torch.randn(1, 3, 224, 224),
    "input_ids": torch.randint(0, 1000, (1, 128)),
    "attention_mask": torch.ones(1, 128),
    "labels": torch.randint(0, 10, (1,)),
    "labels_attention_mask": torch.ones(1, 128)
}

flops = FlopCountAnalysis(model, inputs)
print("FLOPs: {:.2f} GFLOPs".format(flops.total() / 1e9))
print("Parameters: {:.2f} M".format(parameter_count(model)[''] / 1e6))
