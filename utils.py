import torch
from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler
import logging
import asyncio
import uuid

# Configurar el registro
logging.basicConfig(level=logging.INFO)

# Define los estilos y sus rutas correspondientes
model_paths = {
    "cuadros-coleccion": "models/cuadros-coleccion_lora/",
    "cuadros-figuras": "models/cuadros-figuras_lora/",
    "esculturas": "models/esculturas_lora/",
    "esculturas-coleccion": "models/esculturas-coleccion_lora/",
    "murales": "models/murales_lora/"
}

# Define los prompts específicos para cada estilo
style_prompts = {
    "cuadros-coleccion": "in the style of cuadros de coleccion del barroco andino",
    "cuadros-figuras": "in the style of cuadros de figuras del barroco andino",
    "esculturas": "in the style of esculturas del barroco andino",
    "esculturas-coleccion": "in the style of esculturas de coleccion del barroco andino",
    "murales": "in the style of murales del barroco andino"
}

# Determinar el dispositivo a utilizar (GPU si está disponible, de lo contrario CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Cargar el modelo base y configurar los pipelines para cada estilo
pipelines = {}
for style, path in model_paths.items():
    try:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            subfolder="scheduler"
        )
        pipeline.load_lora_weights(path, weight_name="pytorch_lora_weights.safetensors")
        pipelines[style] = pipeline
        logging.info(f"Loaded pipeline for style: {style}")
    except Exception as e:
        logging.error(f"Failed to load pipeline for style {style}: {e}")

# Función para generar imágenes de manera asíncrona
async def generate_image_async(style, custom_prompt):
    if style not in pipelines:
        raise ValueError(f"Estilo '{style}' no está disponible.")
    
    pipeline = pipelines[style]
    prompt = f"{custom_prompt} {style_prompts[style]}"
    logging.info(f"Generating image with prompt: {prompt}")

    # Ajustar parámetros para acelerar la generación
    num_inference_steps = 10  # Reduce el número de pasos de inferencia
    guidance_scale = 7.5
    height = 512
    width = 512

    try:
        loop = asyncio.get_event_loop()
        
        def generate_image():
            image = pipeline(
                prompt, 
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale, 
                height=height, 
                width=width
            ).images[0]
            return image

        current_task = loop.run_in_executor(None, generate_image)
        image = await current_task
        image_id = str(uuid.uuid4())
        image_path = f"/tmp/{image_id}.png"
        image.save(image_path)
        logging.info(f"Image generated and saved to: {image_path}")
        return image_path
    except asyncio.CancelledError:
        logging.info("Image generation task was cancelled.")
        raise
    except Exception as e:
        logging.error(f"Error generating image: {e}")
        raise
