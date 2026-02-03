using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;

namespace PocketTTS
{
    public class MimiEncoder : BackgroundRunner
    {
        public string encoderPath = string.Empty;

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

        private InferenceSession _mimiEncoder;

        public void InitModel()
        {
            if (status != ModelStatus.Init)
                return;

            status = ModelStatus.Loading;
            RunBackground(RunInitModel);
        }

        void RunInitModel(CancellationToken cts)
        {
            try
            {
                var opt = TensorUtil.GetMobileSessionOptions();

                _mimiEncoder = new InferenceSession(encoderPath, opt);

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
            _mimiEncoder?.Dispose();
            _mimiEncoder = null;
        }

        public VoiceInfo Encode(float[] audioData)
        {
            long[] shape = { 1, 1, audioData.Length };
            using var input = OrtValue.CreateTensorValueFromMemory(audioData, shape);
            var inputs = new Dictionary<string, OrtValue>
            {
                { "audio", input }
            };

            using var results = _mimiEncoder.Run(new RunOptions(), inputs, _mimiEncoder.OutputNames);
            var outputValue = results[0];
            var shapeInfo = outputValue.GetTensorTypeAndShape();
            long[] outputDimensions = shapeInfo.Shape;
            var outputSpan = outputValue.GetTensorDataAsSpan<float>();

            return new VoiceInfo
            {
                Data = outputSpan.ToArray(),
                Shape = outputDimensions.Select(d => (long)d).ToArray()
            };
        }
    }
}
