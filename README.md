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
    "ai.lookbe.llamacpp": "https://github.com/lookbe/llama-cpp-unity.git",
    "ai.lookbe.pockettts": "https://github.com/lookbe/pocket-tts-unity.git",

    ... other dependencies
  }
}
```

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
