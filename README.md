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

You must download the following two models separately:

1.  **Orpheus TTS (GGUF format):** e.g., [orpheus-3b-0.1-ft-Q4_K_M-GGUF](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF).
2.  **SNAC Decoder (ONNX format):** You must use the exact `decoder_model.onnx` file from [snac_24khz-ONNX](https://huggingface.co/onnx-community/snac_24khz-ONNX/tree/main/onnx).

---


# Credits

* **[Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)** Developed by **Kyutai Labs** – A TTS that fits in your CPU (and pocket).
    
* **[onnxruntime-unity](https://github.com/asus4/onnxruntime-unity)** Developed by **asus4** – ONNX Runtime integration for the Unity engine.
