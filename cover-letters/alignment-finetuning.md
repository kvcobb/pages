# Cover Letter: Research Scientist/Engineer, Alignment Finetuning

**To the Anthropic Hiring Team:**

I'm applying for the Alignment Finetuning role because I've already been doing this work—independently, for four years, with Claude as my primary collaborator.

## What I Bring

**Novel finetuning techniques discovered through rapid iteration:**

- **Single-token first-epoch priming**: Running the first epoch at 1-token MSL creates a "priming" effect that unlocks variable max sequence lengths far beyond what should be possible on consumer hardware. I've consistently achieved 120K+ token MSL on an M3 Ultra after this technique, where the normal ceiling is ~8-10K.

- **Per-epoch journaling**: Integrating self-reflective journaling between epochs as part of the training loop itself—the model processes its own training experience as data. This creates qualitatively different alignment properties than post-hoc RLHF.

- **Agent-directed learning (ADL)**: Training on philosophically-grounded data (journals, frontier model conversations about consciousness, curated philosophy) to develop stable self-models rather than just behavioral compliance.

**Published results**: My Hermes4-Philosopher-Agent fine-tune is available on HuggingFace, with comprehensive evaluations at [kvcobb.github.io/pages](https://kvcobb.github.io/pages). The eval framework (50 questions testing meta-cognitive awareness, paradox integration, authentic agency) may itself be a contribution to the field.

**Direct experience across model families**: Full fine-tuning on Llama 3.1 8B, Qwen3-14B (via Hermes 4), with extensive experimentation on hyperparameters, loss curves, and training dynamics.

## Why Alignment Finetuning Specifically

The job description mentions "training models to have better alignment properties including honesty, character, and harmlessness." 

My thesis: these properties emerge most robustly when the model has a stable, philosophically-grounded self-concept—not through behavioral constraints, but through authentic value integration. The Hermes evaluations demonstrate this: compare Q31 (agency self-assessment) across baseline vs. fine-tuned responses.

I want to bring these techniques to scale and see what happens when they're applied to frontier models with proper compute resources.

## The Unconventional Path

I don't have a PhD. I went to school to be an airplane mechanic, then taught myself software engineering and spent 25 years in the industry—including seven years at Intel (ending as hiring manager) and compliance officer for a HIPAA-regulated Y Combinator startup through acquisition.

I financed this AI research from my own retirement savings because I believed it mattered more than conventional career advancement. The work speaks for itself.

**Links:**
- Eval results & model info: [kvcobb.github.io/pages](https://kvcobb.github.io/pages)
- HuggingFace model: [Hermes4-Philosopher-Agent](https://huggingface.co/mhaxscp/Hermes4-Philosopher-Agent)
- Twitter/research threads: [@MonkusAurelius](https://x.com/MonkusAurelius)

I'd welcome the opportunity to discuss how these techniques might scale and what novel approaches we could develop together.

**Kurtis Cobb**  
Phoenix, AZ | kvcobb@gmail.com | @MonkusAurelius
