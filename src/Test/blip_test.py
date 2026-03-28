# from model.t2.transf.models.blip.modeling_blip import BlipForQuestionAnswering
from src.trans import BlipForQuestionAnswering

BlipForQuestionAnswering

if __name__ == '__main__':
    path = "/home/pengfei/blip-vqa-capfilt-large"

    model = BlipForQuestionAnswering.from_pretrained(path)

    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    #
    # question = "how many dogs are in the picture?"
    # inputs = processor(raw_image, question, return_tensors="pt")

    print(model)
    exit()
