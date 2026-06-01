"""
SwatchMagic - LoRA test interface
Generate knitting swatches from text prompts using a fine-tuned SD 1.5 LoRA.

Usage:
    1. pip install -r requirements.txt
    2. python app.py  (alt. gradio app.py for live updates)
    3. Open http://127.0.0.1:7860
"""

from pathlib import Path
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pathlib import Path

# ── Paths & model config ─────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
BASE_MODEL   = "runwayml/stable-diffusion-v1-5"
LORA_PATH = BASE_DIR / "training" / "weights" / "lora_weights_run_E" / "checkpoint_epoch5" / "pytorch_lora_weights.safetensors"
TRIGGER_WORD = ""

# ── Generation defaults (used when advanced settings are not changed) ────────
DEFAULT_STEPS      = 25
DEFAULT_GUIDANCE   = 7.5
DEFAULT_LORA_SCALE = 1.0

# ── Device setup ─────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

print(f"Loading {BASE_MODEL} on {device}...")
pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

print(f"Loading LoRA from {LORA_PATH}...")
pipe.load_lora_weights(str(LORA_PATH))
print("Ready.")


# ── "System prompt" equivalents for Stable Diffusion ────────────────────────
PROMPT_PREFIX = (
    "knitting, knitted clothing, textile, yarn texture, fiber craft"
)

NEGATIVE_PROMPT = (
    "animals, dog, cat, food, weather, landscape, architecture, cars, "
    "digital art, painting, illustration, animation, cartoon, text, watermark, "
    "blurry, low quality, distorted"
)

KNITTING_KEYWORDS = {
    "knit", "knitting", "yarn", "wool", "stitch", "swatch", "fiber", "fibre",
    "crochet", "lace", "cable", "ribbed", "rib", "pattern", "weight", "dk",
    "worsted", "aran", "fingering", "bulky", "sport", "colorwork", "fair isle",
    "moss", "seed", "texture", "textile", "fabric", "woven", "cardigan",
    "sweater", "stockinette", "ribbing", "chunky", "stranded", "garter",
    "brioche", "honeycomb", "popcorn",
}


def generate(prompt, steps, guidance, lora_scale, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt describing the swatch.")

    lowered = prompt.lower()
    if not any(kw in lowered for kw in KNITTING_KEYWORDS):
        raise gr.Error(
            "Please describe a knitting swatch — include details like stitch "
            "type, yarn weight, or fibre (e.g. 'cable knit, worsted, cream wool')."
        )

    full_prompt = PROMPT_PREFIX + ", " + prompt.strip()
    if TRIGGER_WORD and TRIGGER_WORD not in full_prompt:
        full_prompt = f"{TRIGGER_WORD}, {full_prompt}"

    generator = None
    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    image = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
        cross_attention_kwargs={"scale": float(lora_scale)},
    ).images[0]
    return image


# ── CSS ──────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
  --pk:    #E85D96;
  --pk-l:  #FBEAF0;
  --pk-m:  #F4C0D1;
  --pk-d:  #993556;
  --or:    #F4733A;
  --or-l:  #FFF5EE;
  --or-d:  #7A3318;
  --text:  #2E2A27;
  --muted: #706B66;
  --hint:  #B5B0AB;
  --rule:  #E8E5E2;
  --surf:  #F9F8F7;
  --white: #FFFFFF;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .gradio-container, .gradio-container > .wrap,
.app, .app > .wrap, .main, .contain {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--white) !important;
  color: var(--text) !important;
}

.gradio-container {
  max-width: 1300px !important;
  margin: 0 auto !important;
  padding: 0 clamp(12px, 3vw, 32px) 40px !important;
}
footer { display: none !important; }

.panel, .form, .block, .gap, .gr-block, .gr-form, .gr-group,
.gr-box, .gr-panel, fieldset, .gr-accordion, .gr-accordion > .label-wrap,
.tab-nav, .tabitem, .tabs, .gr-prose, .gr-padded {
  background: var(--white) !important;
  border-color: var(--rule) !important;
  box-shadow: none !important;
  color: var(--text) !important;
}

/* ── accordion + examples label color fix ── */
.gr-accordion button span,
.gr-accordion button p,
details summary span,
details summary p,
details summary,
.examples-holder > .label,
.examples-holder > label,
.examples > .label,
.examples label,
.label-wrap span,
.label-wrap p,
button.accordion-header,
button.accordion-header span {
  color: #2E2A27 !important;
  font-weight: 500 !important;
  opacity: 1 !important;
}

#topbar {
  height: 3px;
  background: linear-gradient(90deg, var(--pk), var(--or));
  border-radius: 0 0 2px 2px;
  margin-bottom: 0;
}

#app-header {
  background: var(--white) !important;
  border-bottom: 1px solid var(--rule);
  padding: 18px 0 16px;
  margin-bottom: 28px;
}

label > span, .gr-form label span, .block > label > span {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--hint) !important;
}

textarea, input[type="text"], input[type="number"] {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  color: var(--text) !important;
  background: var(--surf) !important;
  border: 1px solid var(--rule) !important;
  border-radius: var(--radius-md) !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
  border-color: var(--pk) !important;
  box-shadow: 0 0 0 3px rgba(232,93,150,0.12) !important;
  background: var(--white) !important;
  outline: none !important;
}

input[type="range"] { accent-color: var(--pk) !important; }
.gr-slider .value-text, input[type="range"] + span {
  background: var(--pk-l) !important;
  color: var(--pk-d) !important;
  border-radius: 20px !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  padding: 2px 8px !important;
  border: none !important;
}

#gen-btn > .wrap > button, #gen-btn button {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  background: linear-gradient(135deg, var(--pk), var(--or)) !important;
  color: var(--white) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 15px 20px !important;
  width: 100% !important;
  cursor: pointer !important;
  transition: opacity 0.15s, transform 0.1s !important;
  box-shadow: 0 2px 12px rgba(232,93,150,0.22) !important;
  letter-spacing: 0.01em !important;
}
#gen-btn > .wrap > button:hover, #gen-btn button:hover {
  opacity: 0.88 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 5px 18px rgba(232,93,150,0.32) !important;
}
#gen-btn > .wrap > button:active, #gen-btn button:active {
  transform: translateY(0) !important;
  opacity: 1 !important;
}

#left-col {
  background: var(--white) !important;
  border: 1px solid var(--rule) !important;
  border-radius: var(--radius-lg) !important;
  padding: clamp(16px, 2.5vw, 24px) !important;
  min-width: 0;
}

#right-col {
  background: var(--surf) !important;
  border: 1px solid var(--rule) !important;
  border-radius: var(--radius-lg) !important;
  padding: clamp(16px, 2.5vw, 24px) !important;
  min-width: 0;
}

#swatch-out, #swatch-out > .wrap, #swatch-out .wrap {
  background: var(--white) !important;
  border: 1px solid var(--rule) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
  min-height: 380px !important;
}
#swatch-out img {
  border-radius: var(--radius-md) !important;
  width: 100% !important;
  height: auto !important;
  display: block !important;
}

.meta-badges { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
.mbadge {
  font-size: 11px;
  font-weight: 500;
  padding: 3px 10px;
  border-radius: 20px;
  background: var(--or-l);
  color: var(--pk);
  border: 1px solid #FFE3CC;
}

.gr-samples-table, .examples-holder table {
  background: var(--white) !important;
  border: 1px solid var(--rule) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
}
.gr-samples-table td, .examples-holder td {
  font-size: 13px !important;
  color: var(--muted) !important;
  padding: 8px 12px !important;
  background: var(--white) !important;
}
.gr-samples-table tr:hover td, .examples-holder tr:hover td {
  background: var(--pk-l) !important;
  color: var(--pk-d) !important;
  cursor: pointer;
}

#main-row {
  display: flex !important;
  gap: clamp(12px, 2vw, 24px) !important;
  align-items: flex-start !important;
  background: transparent !important;
  border: none !important;
  flex-wrap: wrap !important;
}
#left-col  { flex: 1 1 320px; }
#right-col { flex: 1 1 320px; }

input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button { opacity: 0.4; }
"""

# ── HTML fragments ───────────────────────────────────────────────────────────
HEADER = """
<div id="topbar"></div>
<div id="app-header">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <div style="width:32px;height:32px;border-radius:8px;background:#FBEAF0;border:1px solid #F4C0D1;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#E85D96" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 3h18v18H3z"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>
        </svg>
      </div>
      <span style="font-family:'DM Sans',sans-serif;font-size:32px;font-weight:700;color:#E85D96;letter-spacing:-0.02em;">
        ✨SwatchMagic✨
      </span>
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;">
      <span style="font-family:'DM Sans',sans-serif;font-size:11px;padding:3px 10px;border-radius:20px;background:#F9F8F7;color:#706B66;border:1px solid #E8E5E2;font-weight:500;">SD 1.5</span>
      <span style="font-family:'DM Sans',sans-serif;font-size:11px;padding:3px 10px;border-radius:20px;background:#F9F8F7;color:#706B66;border:1px solid #E8E5E2;font-weight:500;">LoRA fine-tuned</span>
      <span style="font-family:'DM Sans',sans-serif;font-size:11px;padding:3px 10px;border-radius:20px;background:#F9F8F7;color:#706B66;border:1px solid #E8E5E2;font-weight:500;">DPM-Solver++</span>
    </div>
  </div>
</div>
"""

META_BADGES = """
<div class="meta-badges">
  <span class="mbadge">512 × 512</span>
  <span class="mbadge">SD 1.5 base</span>
  <span class="mbadge">LoRA fine-tuned</span>
</div>
"""

FOOTER = """
<p style="font-family:'DM Sans',sans-serif;font-size:12px;color:#B5B0AB;text-align:center;padding:14px 0 2px;margin:0;">
  Tip: prompt in comma-separated style matching training captions —
  <em style="color:#D4537E;">cable knit, worsted weight, cream wool</em>
</p>
<p style="font-family:'DM Sans',sans-serif;font-size:12px;color:#B5B0AB;text-align:center;padding:0 0 14px;margin:0;">
  Made by the <a href="https://github.com/HannaKristine00/SwatchMagic" target="_blank" style="color:#D4537E;text-decoration:underline;">SwatchMagic Team</a>
</p>
"""

# ── Layout ───────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="SwatchMagic",
    css=css,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    ).set(
        body_background_fill="#FFFFFF",
        block_background_fill="#FFFFFF",
        block_border_width="0px",
        block_shadow="none",
        panel_background_fill="#FFFFFF",
        button_primary_background_fill="linear-gradient(135deg,#E85D96,#F4733A)",
        button_primary_text_color="white",
        button_primary_border_color="transparent",
        input_background_fill="#F9F8F7",
        input_border_color="#E8E5E2",
        background_fill_primary="#FFFFFF",
        background_fill_secondary="#F9F8F7",
        border_color_primary="#E8E5E2",
        color_accent="#E85D96",
        color_accent_soft="#FBEAF0",
    )
) as demo:

    gr.HTML(HEADER)

    with gr.Row(elem_id="main-row", equal_height=False):

        # ── Controls ─────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="left-col"):
            prompt = gr.Textbox(
                label="Describe your swatch",
                placeholder="e.g. cable knit, worsted weight, cream wool",
                lines=3,
            )

            gen_btn = gr.Button("✦  Generate", variant="primary", elem_id="gen-btn")

            with gr.Accordion("Advanced settings", open=False):
                steps      = gr.Slider(10, 50,  value=DEFAULT_STEPS,      step=1,   label="Steps")
                guidance   = gr.Slider(1,  15,  value=DEFAULT_GUIDANCE,   step=0.5, label="Guidance scale")
                lora_scale = gr.Slider(0.0, 1.5, value=DEFAULT_LORA_SCALE, step=0.05, label="LoRA strength")
                seed       = gr.Number(value=-1, label="Seed  (−1 = random)", precision=0)

            gr.Examples(
                examples=[X^x
                          
                    ["stockinette, aran weight, white"],
                    ["cable stitch swatch, dk weight, grey"],
                    ["ribbing, bulky weight, blue"],
                    ["fair island stranded colorwork, aran weight, navy and red"],
                    ["lace swatch, fingering weight, openwork pattern"],
                ],
                inputs=prompt,
                label="Quick prompts",
            )

        # ── Preview ──────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="right-col"):
            output = gr.Image(
                label="Preview",
                type="pil",
                height=510,
                elem_id="swatch-out",
            )
            gr.HTML(META_BADGES)

    gr.HTML(FOOTER)

    gen_btn.click(
        generate,
        inputs=[prompt, steps, guidance, lora_scale, seed],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
