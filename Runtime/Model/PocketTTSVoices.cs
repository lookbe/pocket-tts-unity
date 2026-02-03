using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using UnityEngine;

namespace PocketTTS
{
    public class PocketTTSVoices : BackgroundRunner
    {
        public MimiEncoder encoder;

        public string voicesBinPath = string.Empty;
        public string voicesRefPath = string.Empty;

        public delegate void StatusChangedDelegate(ModelStatus status);
        public event StatusChangedDelegate OnStatusChanged;

        private ModelStatus _status = ModelStatus.Init;
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

        protected void PostStatus(ModelStatus s)
        {
            unityContext?.Post(_ => status = s, null);
        }

        void OnDestroy()
        {
            BackgroundStopSync();
            FreeModel();
        }

        private Dictionary<string, VoiceInfo> _voices;

        public void InitModel()
        {
            if (status != ModelStatus.Init)
            {
                return;
            }

            status = ModelStatus.Loading;
            StartCoroutine(WaitForEncoderAndInit());
        }

        IEnumerator WaitForEncoderAndInit()
        {
            yield return new WaitUntil(() => encoder.status == ModelStatus.Ready);
            RunBackground(RunInitModel);
        }

        public VoiceInfo GetVoice(string voiceId)
        {
            if (_voices.ContainsKey(voiceId))
            {
                return _voices[voiceId];
            }
            return null;
        }

        void RunInitModel(CancellationToken cts)
        {
            try
            {
                _voices = ParseVoicesBin(voicesBinPath);

                float[] audioData = WavReader.LoadWav(voicesRefPath);
                _voices["custom"] = encoder.Encode(audioData);

                PostStatus(ModelStatus.Ready);
            }
            catch (Exception e)
            {
                Debug.LogError(e);
                FreeModel();
                PostStatus(ModelStatus.Init);
            }
        }

        void FreeModel()
        {
        }

        Dictionary<string, VoiceInfo> ParseVoicesBin(string path)
        {
            var voices = new Dictionary<string, VoiceInfo>();
            using var fs = System.IO.File.OpenRead(path);
            using var reader = new BinaryReader(fs);

            uint numVoices = reader.ReadUInt32();
            for (int i = 0; i < numVoices; i++)
            {
                byte[] nameBytes = reader.ReadBytes(32);
                string name = System.Text.Encoding.ASCII.GetString(nameBytes).TrimEnd('\0').Trim();
                uint numFrames = reader.ReadUInt32();
                uint embDim = reader.ReadUInt32();
                int count = (int)(numFrames * embDim);
                float[] data = new float[count];

                for (int j = 0; j < count; j++)
                {
                    data[j] = reader.ReadSingle();
                }

                voices[name] = new VoiceInfo { Data = data, Shape = new long[] { 1, numFrames, embDim } };
            }
            return voices;
        }

    }

    public class VoiceInfo
    {
        public float[] Data;
        public long[] Shape;
    }
}
