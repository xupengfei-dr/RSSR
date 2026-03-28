import torch
from transformers import BlipImageProcessor, BlipProcessor

from VQALoader_Eath import EarthVQADataset

if __name__ == '__main__':
    ANSWER_VOC = [0, 1, 2, 3, 4, 5, 6, '0%-10%', '10%-20%', '20%-30%', '30%-40%', '40%-50%', '50%-60%', '60%-70%',
                  '70%-80%', '80%-90%', '90%-100%', 'The roads need to be improved, and waters need to be cleaned up',
                  'This is an important traffic area with 3 intersections',
                  'There are residential, educational, park, and agricultural areas', 'Developing',
                  'There are railways',
                  'This is a very important traffic area with 1 intersection, several viaducts, and several bridges',
                  'There are cement roads', 'There are educational, construction, and agricultural areas',
                  'Underdeveloped',
                  'There are unsurfaced roads, and cement roads',
                  'There are residential, commercial, park, and agricultural areas', 'There are commercial areas',
                  'This is a very important traffic area with 2 intersections, and several viaducts',
                  'There are commercial, construction, and park areas',
                  'There are residential, commercial, park, industrial, and agricultural areas',
                  'There are commercial, and construction areas', 'This is not an important traffic area',
                  'This is a very important traffic area with 2 intersections, and several bridges',
                  'There are unsurfaced roads, and railways', 'There are woodland, industrial, and agricultural areas',
                  'There are park areas', 'There are construction, park, and agricultural areas',
                  'There are residential, and industrial areas', 'There are residential, and construction areas',
                  'There is no water', 'There are residential, construction, and park areas',
                  'There are commercial buildings', 'There are agricultural areas', 'There are educational areas',
                  'There are residential, and commercial areas',
                  'There are commercial, educational, park, and industrial areas',
                  'There are clean waters near the agriculture land', 'There are ponds',
                  'There are residential, commercial, park, and industrial areas',
                  'There are educational, park, industrial, and agricultural areas',
                  'There are unsurfaced roads, cement roads, railways, and asphalt roads',
                  'There are one-way lanes, and railways',
                  'There are residential, commercial, educational, park, and industrial areas',
                  'There are no water area',
                  'There are railways, and asphalt roads', 'There are construction areas',
                  'The urban villages need attention', 'There are unsurfaced roads, railways, and asphalt roads',
                  'There are residential, and agricultural areas',
                  'There are residential, commercial, and agricultural areas', 'No',
                  'This is a very important traffic area with 1 intersection, and several viaducts',
                  'The greening needs to be supplemented',
                  'There are residential, commercial, educational, and construction areas',
                  'This is an important traffic area with several bridges',
                  'There are residential, commercial, educational, and industrial areas', 'There are woodland areas',
                  'There are residential, commercial, and construction areas', 'Rural',
                  'There are residential, construction, park, industrial, and agricultural areas',
                  'There are residential, woodland, industrial, and agricultural areas',
                  'This is an important traffic area with 4 intersections', 'There are private buildings',
                  'There are woodland, and agricultural areas',
                  'There are residential, commercial, construction, and park areas', 'There are rivers and ponds',
                  'There are residential, construction, and agricultural areas',
                  'There are residential, and educational areas', 'There are commercial, and educational areas',
                  'There are polluted waters near the agriculture land',
                  'There are one-way lanes, wide lanes, and railways', 'There are one-way lanes, and wide lanes',
                  'Urban',
                  'There are residential, commercial, and educational areas', 'There are commercial, and park areas',
                  'There are unsurfaced roads, cement roads, and asphalt roads',
                  'There are commercial buildings, and private buildings',
                  'This is an important traffic area with 1 intersection',
                  'There are commercial, industrial, and agricultural areas',
                  'There are residential, commercial, construction, park, and industrial areas',
                  'There are asphalt roads',
                  'There are residential, commercial, and park areas', 'There are no agricultural land',
                  'There are commercial, construction, park, and agricultural areas',
                  'There are residential, educational, and construction areas',
                  'There are commercial, construction, and industrial areas',
                  'There are residential, commercial, construction, and industrial areas',
                  'There are park, and industrial areas', 'There are commercial, and agricultural areas',
                  'There are residential, educational, construction, and park areas', 'No obvious land use types',
                  'There are construction, park, and industrial areas',
                  'There are residential, educational, park, and industrial areas',
                  'There are commercial, park, and industrial areas',
                  'This is an important traffic area with several viaducts',
                  'This is a very important traffic area with 1 intersection, and several bridges',
                  'There are residential, park, and agricultural areas',
                  'There are residential, commercial, construction, and agricultural areas',
                  'There are residential, commercial, educational, construction, park, and agricultural areas',
                  'There are wide lanes, and railways', 'There are residential, park, and industrial areas',
                  'There are residential, industrial, and agricultural areas', 'There are construction, and park areas',
                  'There are residential, commercial, construction, park, industrial, and agricultural areas',
                  'There are residential, park, industrial, and agricultural areas', 'There are residential areas',
                  'There are residential, commercial, educational, park, and agricultural areas',
                  'There are residential, commercial, industrial, and agricultural areas',
                  'There are residential, commercial, educational, and park areas',
                  'There are construction, and agricultural areas', 'There are no water nor agricultural land',
                  'The waters need to be cleaned up', 'There are park, and agricultural areas', 'There are rivers',
                  'This is a very important traffic area with 3 intersections, and several viaducts',
                  'This is an important traffic area with 2 intersections', 'There are industrial areas',
                  'There are unsurfaced roads, and asphalt roads',
                  'This is a very important traffic area with 2 intersections, several viaducts, and several bridges',
                  'There are commercial, park, and agricultural areas', 'There are one-way lanes',
                  'There are residential, educational, construction, and agricultural areas', 'There are no roads',
                  'There are residential, construction, park, and agricultural areas',
                  'There are residential, and park areas', 'There are commercial, construction, and agricultural areas',
                  'There are cement roads, and asphalt roads',
                  'There are residential, educational, and agricultural areas',
                  'There are commercial, and industrial areas', 'There are park, industrial, and agricultural areas',
                  'This is a very important traffic area with several viaducts, and several bridges',
                  'There are educational, construction, and park areas',
                  'There are residential, woodland, and agricultural areas',
                  'There are residential, and woodland areas',
                  'There are unsurfaced roads, cement roads, and railways',
                  'There are educational, park, and agricultural areas',
                  'There are residential, educational, and park areas',
                  'There are commercial, educational, and park areas',
                  'There are wide lanes', 'There are cement roads, and railways', 'There are no residential buildings',
                  'There are commercial, park, industrial, and agricultural areas',
                  'There are residential, commercial, and industrial areas',
                  'The greening needs to be supplemented and urban villages need attention', 'There is no barren land',
                  'There are educational, and agricultural areas', 'The roads need to be improved', 'Yes',
                  'There are unsurfaced roads',
                  'There are residential, commercial, construction, park, and agricultural areas',
                  'There are residential, construction, and industrial areas',
                  'There are cement roads, railways, and asphalt roads', 'There are educational, and park areas',
                  'There are no needs']
    QUESTIONS = ['Are there any villages in this scene?', 'Is there a commercial area near the residential area?',
                 'Are there any playgrounds in this scene?', 'Is there any commercial land in this scene?',
                 'Is there any forest in this scene?', 'Is there any agriculture in this scene?',
                 'What are the types of residential buildings?', 'Are there any urban villages in this scene?',
                 'What are the needs for the renovation of villages?', 'Is there any barren in this scene?',
                 'Whether greening need to be supplemented in residential areas?',
                 'Is there any woodland in this scene?',
                 'What are the land use types in this scene?', 'What are the needs for the renovation of residents?',
                 'Are there any buildings in this scene?', 'Is there any agricultural land in this scene?',
                 'What is the area of roads?', 'What is the area of playgrounds?', 'Is it a rural or urban scene?',
                 'What is the area of barren?', 'Are there any bridges in this scene?',
                 'Are there any eutrophic waters in this scene?', 'Are there any viaducts in this scene?',
                 'What is the area of water?', 'Are there any roads in this scene?',
                 'Is there any residential land in this scene?', 'How many eutrophic waters are in this scene?',
                 'Is there any industrial land in this scene?', 'Is there any park land in this scene?',
                 'Is there any uncultivated agricultural land in this scene?',
                 'Is there a school near the residential area?',
                 'Are there any large driveways (more than four lanes)?',
                 'What are the comprehensive traffic situations in this scene?', 'What is the area of buildings?',
                 'Is there any construction land in this scene?', 'What are the water types in this scene?',
                 'Are there any viaducts near the residential area?',
                 'Is there a construction area near the residential area?',
                 'Is there a park near the residential area?',
                 'What are the road materials around the village?', 'Are there any intersections in this scene?',
                 'What are the road types around the residential area?',
                 'What are the water situations around the agricultural land?', 'What is the situation of barren land?',
                 'What is the area of the forest?', 'Are there any intersections near the school?',
                 'Is there any water in this scene?', 'Is there any educational land in this scene?',
                 'How many intersections are in this scene?', 'Are there any greenhouses in this scene?',
                 'What is the area of agriculture?']
    # --- 初始化处理器和分词器 ---
    num_workers = 8
    image_processor = BlipImageProcessor(do_resize=True, image_std=[0.229, 0.224, 0.225],
                                         image_mean=[0.485, 0.456, 0.406], do_rescale=True, do_normalize=True,
                                         size=384, size_divisor=32)
    blip_processor_path = "/home/gpuadmin/blip-vqa-base"
    try:
        processor = BlipProcessor.from_pretrained(blip_processor_path)
        tokenizer = processor.tokenizer
    except Exception as e:
        print(f"Error loading BlipProcessor from '{blip_processor_path}': {e}")

    train_dataset = EarthVQADataset(
        json_path='/home/gpuadmin/DataSet/EathVQA/Train_QA.json',
        img_folder='/home/gpuadmin/DataSet/EathVQA/Train/images_png',
        answer_voc=ANSWER_VOC,
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    LR_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )

    from itertools import islice
    import torch


    def summarize_tensor(t: torch.Tensor):
        return f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, min={t.min().item():.3f}, max={t.max().item():.3f}"


    def print_batch(bi: int, batch: dict, sample_idx: int = 0):
        print(f"\n=== Batch #{bi} ===")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # 尽量打印张量概要 + 一个样本的形状
                info = summarize_tensor(v)
                one_item_shape = tuple(v[sample_idx].shape) if v.ndim > 0 else ()
                print(f"{k:>22s}: {info}, one_item_shape={one_item_shape}")
            elif isinstance(v, (list, tuple)):
                ln = len(v)
                elem = v[sample_idx] if ln > 0 else None
                tname = type(elem).__name__ if elem is not None else "None"
                print(f"{k:>22s}: list(len={ln}, elem_type={tname}), sample_0={elem}")
            else:
                print(f"{k:>22s}: {type(v).__name__} -> {v}")

        # 打印一些关键字段的前几个值（如果存在）
        for key in ["q_idx", "image_n", "question_type", "labels", "answer"]:
            if key in batch:
                v = batch[key]
                if isinstance(v, torch.Tensor):
                    print(f"{key:>22s} head: {v[:8].tolist()}")
                elif isinstance(v, (list, tuple)):
                    print(f"{key:>22s} head: {list(v)[:8]}")
                else:
                    print(f"{key:>22s}: {v}")


    # 取并打印前3个批次
    for bi, batch in enumerate(islice(LR_train_loader, 3)):
        # 如果用了 pin_memory=True，打印前先转到 CPU，避免某些环境下的显示问题
        batch = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        print_batch(bi, batch)
