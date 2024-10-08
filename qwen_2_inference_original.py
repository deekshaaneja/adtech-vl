from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
#import torch
#torch.manual_seed(1234)

import argparse

parser=argparse.ArgumentParser()

parser.add_argument("--image", help="Path of the image")

args=parser.parse_args()


def infer(url, q_list):
    responses = []
    # Note: The default behavior now has injection attack prevention off.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


    # 1st dialogue turn
    for q in q_list:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": url,
                    },
                    {"type": "text", "text": q},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        responses.append(output_text)
    return responses


if __name__ == '__main__':
    print(f"Predicting for image: {args.image}")
    url = args.image
    q1 = "Give me the title (in 25 characters) for personalized advertisement"
    q2 = "Give me the description (in 90 characters) for personalized advertisement"
    q3 = "Give me the keywords for advertisement of this product"
    print(infer(url, [q1, q2, q3]))




