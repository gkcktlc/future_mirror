import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import whisper
from transformers import pipeline


pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token=True
).to("cuda" if torch.cuda.is_available() else "cpu")

whisper_model = whisper.load_model("base")

#  GPT-2 prompt zenginleştirme
generator = pipeline("text-generation", model="gpt2")

def enrich_prompt(prompt):
    instruction = f"Describe in more detail, like it's a futuristic sci-fi vision: {prompt}"
    result = generator(instruction, max_length=80, do_sample=True, temperature=0.9)[0]['generated_text']
    return result

#  SES/YAZI 
def generate_image_combined(input_type, text_input, audio_input):
    if input_type == "📝 Text":
        prompt = text_input
    else:
        result = whisper_model.transcribe(audio_input)
        prompt = result["text"]

    prompt = enrich_prompt(prompt)
    image = pipe(prompt).images[0]
    image.save("output.png")
    return image, "output.png", prompt

#  ARAYÜZ (görsel kutusu "ayna efekti" ile süslenmeye çalışdı ama olmadı)
with gr.Blocks(css="""
.gr-image {
    border: 4px solid rgba(255, 255, 255, 0.3);
    border-radius: 20px;
    box-shadow:
        inset 0 0 60px rgba(255, 255, 255, 0.15),
        0 0 30px rgba(255, 255, 255, 0.05),
        0 0 10px rgba(255, 255, 255, 0.05);
    background: linear-gradient(to bottom right, #1a1a1a, #2d2d2d);
    backdrop-filter: blur(6px);
    padding: 8px;
    transition: all 0.3s ease-in-out;
}
""") as demo:

    gr.Markdown("# 🔮 Future Mirror AI")
    gr.Markdown("Type or speak your futuristic vision — and see what it might look like in 2045!")

    with gr.Row():
        with gr.Column(scale=1):
            input_type = gr.Radio(["📝 Text", "🎤 Voice"], label="Select Input Type", value="📝 Text")

            text_input = gr.Textbox(
                label="Type your description",
                placeholder="e.g., a robot turtle flying over a neon city",
                lines=2
            )

            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Speak your idea"
            )

            generate_btn = gr.Button("✨ Generate")
            output_file = gr.File(label="⬇️ Download Image")

        with gr.Column(scale=2):
            output_img = gr.Image(label="🪞 Magic Mirror Output", type="pil")
            prompt_out = gr.Textbox(label="📝 Used Prompt", interactive=False)

    generate_btn.click(
        fn=generate_image_combined,
        inputs=[input_type, text_input, audio_input],
        outputs=[output_img, output_file, prompt_out]
    )

# 1. app.py
code = '''# Your full Gradio project code goes here
# paste everything from `import gradio as gr` to demo.launch(share=True)'''
with open("app.py", "w") as f:
    f.write(code)

# 2. requirements.txt
reqs = '''gradio
torch
diffusers
transformers
accelerate
whisper
ffmpeg-python'''
with open("requirements.txt", "w") as f:
    f.write(reqs)

# 3. README.md
readme = '''# Future Mirror AI 🪞
This app allows you to describe or speak your imagination,
and see what it might look like in 2045 — visualized through a magical mirror.

## Features
- Text or voice input
- Whisper for speech-to-text
- Stable Diffusion for image generation
- Gradio UI with mirror frame overlay

## Run Instructions

Developed by Gökçe Kutluca.
'''
with open("README.md", "w") as f:
    f.write(readme)


# zıp 
!zip -r future-mirror-ai.zip app.py requirements.txt README.md mirror_frame_ready.png




demo.launch(share=True)
