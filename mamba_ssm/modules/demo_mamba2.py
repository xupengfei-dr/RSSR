from transformers import MambaForCausalLM, Mamba2ForCausalLM, Mamba2Model, AutoModel

if __name__ == '__main__':
    model = Mamba2Model.from_pretrained("/home/pengfei/mamba2-130m")
    print(model)
