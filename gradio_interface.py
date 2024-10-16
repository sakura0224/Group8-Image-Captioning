import gradio as gr
from inference import generate_caption_from_image

# Gradio 处理函数


def predict_caption(image):
    caption = generate_caption_from_image(image)  # 调用 inference.py 中的函数
    return caption

# def predict_caption(image):
#     print("图片接收成功。")  # 确认图片已上传
#     return "这是一张图片的描述。"


# 创建 Gradio 接口
image_input = gr.Image(type="pil", label="上传图片")
text_output = gr.Textbox(label="生成的描述")

# 启动 Gradio 接口，并启用共享链接
gr.Interface(fn=predict_caption, inputs=image_input, outputs=text_output,
             title="图像描述生成", description="上传一张图片，生成图片描述。").launch(share=True, server_port=7000)
