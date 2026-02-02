using LlamaCpp;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace PocketTTS
{
    public class PocketTTS : MonoBehaviour
    {
        public enum Voice
        {
            cosette,
            jean,
            fantine,
            custom,
        }

        public Voice voice = Voice.cosette;

        [Header("Voices")]
        public string voicesBinPath = string.Empty;
        public string voicesRefPath = string.Empty;

        [Header("Codec")]
        public string encoderPath = string.Empty;
        public string decoderPath = string.Empty;

        [Header("Text")]
        public string tokenizerPath = string.Empty;
        public string textConditionerPath = string.Empty;

        [Header("Models")]
        public string flowLmMainPath = string.Empty;
        public string flowLmFlowPath = string.Empty;

        protected PocketTTSModel pocketTTS;
        protected PocketTTSVoices voices;
        protected MimiEncoder encoder;
        protected MimiDecoder decoder;
        protected AudioSource audioSource;

        private Queue<float> audioQueue = new Queue<float>();

        public delegate void StatusChangedDelegate(ModelStatus status);
        public event StatusChangedDelegate OnStatusChanged;

        private ModelStatus _status = ModelStatus.Init;

        // Public getter, no public setter
        public ModelStatus status
        {
            get => _status;
            protected set
            {
                if (_status != value)
                {
                    _status = value;
                    OnStatusChanged?.Invoke(_status);
                }
            }
        }

        // harcoded value from snac decoder
        private const int SampleRate = 24000;
        private const int Channels = 1;

        private void Awake()
        {
            pocketTTS = GetComponentInChildren<PocketTTSModel>();
            voices = GetComponentInChildren<PocketTTSVoices>();
            encoder = GetComponentInChildren<MimiEncoder>();
            decoder = GetComponentInChildren<MimiDecoder>();
            audioSource = GetComponent<AudioSource>();

            audioSource.clip = AudioClip.Create("StreamingClip", SampleRate * 60, Channels, SampleRate, true, OnAudioRead);
            audioSource.loop = true;
            audioSource.Play();
        }

        private void OnEnable()
        {
            pocketTTS.OnStatusChanged += OnModelStatusChanged;
            voices.OnStatusChanged += OnModelStatusChanged;
            decoder.OnStatusChanged += OnModelStatusChanged;
            encoder.OnStatusChanged += OnModelStatusChanged;

            pocketTTS.OnResponseGenerated += OnResponseGenerated;
        }

        private void OnDisable()
        {
            pocketTTS.OnStatusChanged -= OnModelStatusChanged;
            voices.OnStatusChanged -= OnModelStatusChanged;
            decoder.OnStatusChanged -= OnModelStatusChanged;
            encoder.OnStatusChanged -= OnModelStatusChanged;

            pocketTTS.OnResponseGenerated -= OnResponseGenerated;
        }

        void OnModelStatusChanged(ModelStatus status)
        {
            if (status == ModelStatus.Error)
            {
                StopAllCoroutines();
                this.status = ModelStatus.Error;
            }
        }

        void OnResponseGenerated(float[] audioChunk)
        {
            if (audioChunk == null || audioChunk.Length == 0)
            {
                return;
            }

            try
            {
                foreach (var s in audioChunk)
                    audioQueue.Enqueue(s);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Queue failure: {ex.Message}");
            }
        }

        private void OnAudioRead(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                if (audioQueue.Count > 0)
                    data[i] = audioQueue.Dequeue();
                else
                    data[i] = 0f;
            }
        }

        public void InitModel()
        {
            if (string.IsNullOrEmpty(voicesBinPath))
            {
                return;
            }

            if (string.IsNullOrEmpty(encoderPath) || string.IsNullOrEmpty(decoderPath))
            {
                return;
            }

            if (string.IsNullOrEmpty(tokenizerPath) || string.IsNullOrEmpty(textConditionerPath))
            {
                return;
            }

            if (string.IsNullOrEmpty(flowLmMainPath) || string.IsNullOrEmpty(flowLmFlowPath))
            {
                return;
            }

            if (_status != ModelStatus.Init)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Loading;
            StartCoroutine(RunInitModel());
        }

        IEnumerator RunInitModel()
        {
            Debug.Log($"Load tts model");

            pocketTTS.tokenizerPath = tokenizerPath;
            pocketTTS.textConditionerPath = textConditionerPath;
            pocketTTS.flowLmMainPath = flowLmMainPath;
            pocketTTS.flowLmFlowPath = flowLmFlowPath;
            pocketTTS.InitModel();

            encoder.encoderPath = encoderPath;
            encoder.InitModel();

            decoder.decoderPath = decoderPath;
            decoder.InitModel();

            voices.voicesBinPath = voicesBinPath;
            voices.voicesRefPath = voicesRefPath;
            voices.InitModel();

            yield return new WaitWhile(() => pocketTTS.status != ModelStatus.Ready);
            yield return new WaitWhile(() => encoder.status != ModelStatus.Ready);
            yield return new WaitWhile(() => decoder.status != ModelStatus.Ready);
            yield return new WaitWhile(() => voices.status != ModelStatus.Ready);

            Debug.Log("Load model done");

            status = ModelStatus.Ready;
        }

        public void Prompt(string prompt)
        {
            if (string.IsNullOrEmpty(prompt))
            {
                return;
            }

            if (status != ModelStatus.Ready)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Generate;
            StartCoroutine(WaitForGenerationAndPlaybackDone(prompt));
        }

        IEnumerator WaitForGenerationAndPlaybackDone(string prompt)
        {
            pocketTTS.PromptWithVoice(prompt, voices.GetVoice(voice.ToString()));

            yield return new WaitUntil(() => pocketTTS.status == ModelStatus.Ready);
            yield return new WaitUntil(() => decoder.status == ModelStatus.Ready);

            // Wait for all audio samples to be played
            yield return new WaitUntil(() => audioQueue.Count == 0);

            status = ModelStatus.Ready;
        }

        public void Stop()
        {
            if (status != ModelStatus.Generate)
            {
                Debug.Log("already stopped");
                return;
            }

            //pocketTTS.Stop();
        }
    }
}
