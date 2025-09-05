from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image


model, processor = load("mlx-community/Qwen2-VL-7B-Instruct-8bit")
config = model.config

image_path = "images/test.jpg"
image = load_image(image_path)

messages = [
    {
        "role": "user",
        "content": """detect all the objects in the image, return bounding boxes for all of them using the following format: [{
        "object": "object_name",
        "bboxes": [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
     }, ...]""",
    }
]
prompt = apply_chat_template(processor, config, messages)

output = generate(model, processor, image, prompt, max_tokens=1000, temperature=0.7)
print(output)