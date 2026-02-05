# Pocket TTS for Unity

A Unity 6 integration for [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts).

---

## ⚠️ Hardware & Platform Support
* **Platform:** **Windows** & **Android**.

---

## Installation

Follow these steps exactly to ensure all native dependencies are resolved.

### Configure manifest.json
Open your project's `Packages/manifest.json` and update it to include the scoped registry and the Git dependencies.


```json
{
  "scopedRegistries": [
    {
      "name": "npm",
      "url": "https://registry.npmjs.com",
      "scopes": [
        "com.github.asus4"
      ]
    }
  ],
  "dependencies": {
    "com.github.asus4.onnxruntime": "0.4.2",
    "com.github.asus4.onnxruntime.unity": "0.4.2",
    "ai.lookbe.pockettts": "https://github.com/lookbe/pocket-tts-unity.git",

    ... other dependencies
  }
}
```

---

## Mobile Tuning Guide

Running this model on mobile devices requires balancing **Performance (Stuttering)** vs. **Latency** vs. **Quality**. Use the `[PocketTTS Stats]` log in the console to profile your device.

The logs will show execution times:
`[TTS Stats] AR: 50ms | Flow: 20ms | Mimi: 300ms`

Also watch for the Real-Time Ratio (RTFx):
`[TTS RT] Ratio: 0.70x (LAGGING)`

### Tuning Parameters
You can adjust these settings on the `PocketTTSModel` component:

#### 1. Diffusion Step (Default: 10)
Controlled by `DiffusionStep`. This is the most impactful setting for performance.
* **Lower (e.g., 2-4):**
  * ✅ Drastically reduces computation per frame.
  * ✅ Fixes stuttering on weaker devices.
  * ❌ **Lower voice quality**: Speech may sound more robotic or have artifacts.
* **Higher (e.g., 10+):**
  * ✅ High-quality, natural speech.
  * ❌ Heavy CPU usage.

#### 2. Audio Chunk Size (Default: 12)
Controlled by `AudioChunkSize`. This determines how many frames are generated before decoding audio.
* **Larger (e.g., 16-24):**
  * ✅ **Better throughput**: Amortizes the heavy overhead of the Mimi Decoder.
  * ❌ **Increased Latency**: Users wait longer to hear the *first* word of a sentence.
* **Smaller (e.g., 4-8):**
  * ✅ **Fast response**: Good for conversational agents.
  * ❌ **Stuttering Risk**: The overhead of frequent decoder calls may cause the pipeline to fall behind real-time.

### Example Profiles

**Mid-Range Device (e.g., Helio G99):**
* **Goal:** Smooth playback, acceptable latency.
* **Settings:** `DiffusionStep: 3`, `AudioChunkSize: 16`
* **Result:** No stuttering, ~1.4s startup latency.

**High-End Device (e.g., Snapdragon 8 Gen 3):**
* **Goal:** Maximum quality.
* **Settings:** `DiffusionStep: 10`, `AudioChunkSize: 8`
* **Result:** High-fidelity speech, fast response.

---

## Requirements: Models

### A. ONNX ([KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx/tree/main))
* `mimi_encoder.onnx`
* `mimi_decoder.onnx`
* `text_conditioner.onnx`
* `flow_lm_main.onnx`
* `flow_lm_flow.onnx`
* `tokenizer.model`

### B. Predefined Voices ([KevinAHM/pocket-tts-web](https://huggingface.co/spaces/KevinAHM/pocket-tts-web/tree/main))
* `voices.bin`

---


# Credits

* **[Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)** Developed by **Kyutai Labs** – A TTS that fits in your CPU (and pocket).
    
* **[onnxruntime-unity](https://github.com/asus4/onnxruntime-unity)** Developed by **asus4** – ONNX Runtime integration for the Unity engine.
