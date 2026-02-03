using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Threading;
using Unity.VisualScripting;
using UnityEngine;

namespace PocketTTS
{
    public class MimiDecoder : BackgroundRunner
    {
        public string decoderPath = string.Empty;

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

        private InferenceSession _mimiDecoder;

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
                var opt = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                opt.IntraOpNumThreads = 4;

                _mimiDecoder = new InferenceSession(decoderPath, opt);

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
            _mimiDecoder?.Dispose();
            _mimiDecoder = null;
        }

        public float[] DecodeChunk(List<float[]> chunk, ref Dictionary<string, OrtValue> state)
        {
            try
            {
                int count = chunk.Count;
                float[] flat = new float[count * 32];
                for (int i = 0; i < count; i++)
                {
                    Array.Copy(chunk[i], 0, flat, i * 32, 32);
                }

                using var latent = OrtValue.CreateTensorValueFromMemory<float>(OrtMemoryInfo.DefaultInstance, flat, new long[] { 1, count, 32 });
                using var res = _mimiDecoder.Run(new RunOptions(), TensorUtil.Merge(state, ("latent", latent)), _mimiDecoder.OutputNames);

                var audioOutput = res[0];
                ReadOnlySpan<float> audioSpan = audioOutput.GetTensorDataAsSpan<float>();

                TensorUtil.UpdateState(state, res, _mimiDecoder.OutputNames);

                return audioSpan.ToArray();
            }
            catch (Exception e)
            {
                Debug.LogError(e);
            }

            return new float[0];
        }

        public Dictionary<string, OrtValue> InitMimiDecoderState()
        {
            var state = new Dictionary<string, OrtValue>();

            state["state_0"] = TensorUtil.CreateBoolTensor(false);
            state["state_1"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 512, 6 });
            state["state_2"] = TensorUtil.CreateBoolTensor(false);
            state["state_3"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 64, 2 });
            state["state_4"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 256, 6 });
            state["state_5"] = TensorUtil.CreateBoolTensor(false);
            state["state_6"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 256, 2 });
            state["state_7"] = TensorUtil.CreateBoolTensor(false);
            state["state_8"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 128, 0 });
            state["state_9"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 128, 5 });
            state["state_10"] = TensorUtil.CreateBoolTensor(false);
            state["state_11"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 128, 2 });
            state["state_12"] = TensorUtil.CreateBoolTensor(false);
            state["state_13"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 64, 0 });
            state["state_14"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 64, 4 });
            state["state_15"] = TensorUtil.CreateBoolTensor(false);
            state["state_16"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 64, 2 });
            state["state_17"] = TensorUtil.CreateBoolTensor(false);
            state["state_18"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 32, 0 });
            state["state_19"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 2, 1, 8, 1000, 64 });
            state["state_20"] = TensorUtil.CreateInt64Tensor(0);
            state["state_21"] = TensorUtil.CreateInt64Tensor(0);
            state["state_22"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 2, 1, 8, 1000, 64 });
            state["state_23"] = TensorUtil.CreateInt64Tensor(0);
            state["state_24"] = TensorUtil.CreateInt64Tensor(0);
            state["state_25"] = TensorUtil.CreateBoolTensor(false);
            state["state_26"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 512, 16 });
            state["state_27"] = TensorUtil.CreateBoolTensor(false);
            state["state_28"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 1, 6 });
            state["state_29"] = TensorUtil.CreateBoolTensor(false);
            state["state_30"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 64, 2 });
            state["state_31"] = TensorUtil.CreateBoolTensor(false);
            state["state_32"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 32, 0 });
            state["state_33"] = TensorUtil.CreateBoolTensor(false);
            state["state_34"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 512, 2 });
            state["state_35"] = TensorUtil.CreateBoolTensor(false);
            state["state_36"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 64, 4 });
            state["state_37"] = TensorUtil.CreateBoolTensor(false);
            state["state_38"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 128, 2 });
            state["state_39"] = TensorUtil.CreateBoolTensor(false);
            state["state_40"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 64, 0 });
            state["state_41"] = TensorUtil.CreateBoolTensor(false);
            state["state_42"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 128, 5 });
            state["state_43"] = TensorUtil.CreateBoolTensor(false);
            state["state_44"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 256, 2 });
            state["state_45"] = TensorUtil.CreateBoolTensor(false);
            state["state_46"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 128, 0 });
            state["state_47"] = TensorUtil.CreateBoolTensor(false);
            state["state_48"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 256, 6 });
            state["state_49"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 2, 1, 8, 1000, 64 });
            state["state_50"] = TensorUtil.CreateInt64Tensor(0);
            state["state_51"] = TensorUtil.CreateInt64Tensor(0);
            state["state_52"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 2, 1, 8, 1000, 64 });
            state["state_53"] = TensorUtil.CreateInt64Tensor(0);
            state["state_54"] = TensorUtil.CreateInt64Tensor(0);
            state["state_55"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 512, 16 });

            return state;
        }
    }
}
