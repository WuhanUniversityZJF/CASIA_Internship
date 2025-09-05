import json
from pycocotools.coco import COCO

#from mmengine.dataset.base_dataset import Compose
from mmengine.dataset import Compose  # 注意：从 mmengine.dataset 导入 Compose

from mmdet.registry import DATASETS
from PIL import Image



# -------------------------- 新增：注册 MMDetection 所有模块 --------------------------
from mmdet.utils.setup_env import register_all_modules
# 显式注册 MMDetection 内置模块（包括 PackDetInputs）
register_all_modules(True)  # True 确保覆盖已有注册，避免冲突
# -----------------------------------------------------------------------------------


def create_annfile(file_name):
    save_ann = {
        'categories': [{'supercategory': 'chat', 'id': 1, 'name': 'chat'}],
        'annotations': [],
        'images': [{'file_name': file_name, 'id': 1},]
    }
    return save_ann


def get_data_info(image_info, coco_anns, text):
    image = Image.open(image_info['file_name']).convert("RGB")
    data_info = {}
    data_info.update({
        'img_path': image_info['file_name'],
        'img_id': image_info['id'],
        'seg_map_path': None,
        'height': image.height,
        'width': image.width,
        'text': text,
        'custom_entities': True, # whether return classes
        'instances': [],
    })
    
    return data_info

def load_as_mmdet(file_name, caption=None):
    assert caption != None
    text = tuple([caption])
    mmdet_pipline_cfg = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='Resize', scale=(800, 1333), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'text', 'custom_entities'))
    ]
    mmdet_pipeline = Compose(mmdet_pipline_cfg)
    ann_file = create_annfile(file_name)

    g_dino_data_info = get_data_info(ann_file['images'][0], ann_file['annotations'], text)
    g_dino_data = mmdet_pipeline(g_dino_data_info)

    return g_dino_data
