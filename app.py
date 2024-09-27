import gradio as gr
#import spaces
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
import devicetorch
from PIL import Image

MODELS = {
    "RealVisXL V5.0 Lightning": "SG161222/RealVisXL_V5.0_Lightning",
}
DEVICE = devicetorch.get(torch)
pipe = None
global_image = None
def init():
    global pipe

    if pipe is None:

        config_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="config_promax.json",
        )

        config = ControlNetModel_Union.load_config(config_file)
        controlnet_model = ControlNetModel_Union.from_config(config)
        model_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="diffusion_pytorch_model_promax.safetensors",
        )
        state_dict = load_state_dict(model_file)
        model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
            controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
        )
        model.to(device=DEVICE, dtype=torch.float16)

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        ).to(DEVICE)

        pipe = StableDiffusionXLFillPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0_Lightning",
            torch_dtype=torch.float16,
            vae=vae,
            controlnet=model,
            variant="fp16",
        ).to(DEVICE)

        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)


#@spaces.GPU(duration=16)
def fill_image(prompt, image, model_selection):
    init()
    source = image["background"]
    mask = image["layers"][0]

    alpha_channel = mask.split()[3]
    binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
    cnet_image = source.copy()
    cnet_image.paste(0, (0, 0), binary_mask)

    print(f"image {image}, cnet_image={cnet_image}")
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(prompt, DEVICE, True)


    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
    ):
        yield image, cnet_image

    print(f"AFTER1 image {image}, cnet_image={cnet_image}")
    image = image.convert("RGBA")
    print(f"AFTER2 image {image}, cnet_image={cnet_image}")
    cnet_image.paste(image, (0, 0), binary_mask)

    yield source, cnet_image


def clear_result():
    return gr.update(value=None)

def resize(image, size):
    global global_image
    size = int(size)
    if global_image is None:
        global_image = image["background"]
    source = global_image.copy()
    print(f"source image={source}")
    source.thumbnail((size, size), Image.LANCZOS)
    print(f"resized image={source}")
    w, h = global_image.size
#            canvas_size=(1024, 1024),
    
    max = (w // 8) * 8
    return gr.update(value=source, canvas_size=(w,h)), gr.update(maximum=max, visible=True)


#css = """
#.gradio-container {
#    width: 1024px !important;
#}
#"""


#with gr.Blocks(css=css, fill_width=True) as demo:
with gr.Blocks(fill_width=True) as demo:
    with gr.Row():
        prompt = gr.Textbox(value="high quality", label="Prompt")
        size = gr.Slider(value=1024, label="Resize", minimum=0, maximum=1024, step=8, visible=False, interactive=True)
        run_button = gr.Button("Generate")

    with gr.Row():
        input_image = gr.ImageMask(
            type="pil",
            label="Input Image",
#            crop_size=(1024, 1024),
#            canvas_size=(1024, 1024),
            layers=False,
            sources=["upload"],
        )

        result = ImageSlider(
            interactive=False,
            label="Generated Image",
        )

    model_selection = gr.Dropdown(
        choices=list(MODELS.keys()),
        value="RealVisXL V5.0 Lightning",
        label="Model",
    )

    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=fill_image,
        inputs=[prompt, input_image, model_selection],
        outputs=result,
    )
    input_image.upload(fn=resize, inputs=[input_image, size], outputs=[input_image, size])
    size.change(fn=resize, inputs=[input_image, size], outputs=[input_image, size])


demo.launch(share=False)
