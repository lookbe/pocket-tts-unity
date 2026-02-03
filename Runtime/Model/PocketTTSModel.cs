using Microsoft.ML.OnnxRuntime;
using PocketTts;
using System;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

namespace PocketTTS
{
    public class PocketTTSModel : BackgroundRunner
    {
        public MimiDecoder decoder;

        [Header("Models")]
        public string tokenizerPath = string.Empty;
        public string textConditionerPath = string.Empty;
        public string flowLmMainPath = string.Empty;
        public string flowLmFlowPath = string.Empty;

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

        private InferenceSession _textConditioner;
        private InferenceSession _flowLmMain;
        private InferenceSession _flowLmFlow;
        private SentencePieceWrapper _sentencePiece;

        #region Init / Free

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
                _sentencePiece = new SentencePieceWrapper(tokenizerPath);

                var opt = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                opt.IntraOpNumThreads = 4;

                _textConditioner = new InferenceSession(textConditionerPath, opt);
                _flowLmMain = new InferenceSession(flowLmMainPath, opt);
                _flowLmFlow = new InferenceSession(flowLmFlowPath, opt);

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
            _textConditioner?.Dispose();
            _textConditioner = null;

            _flowLmMain?.Dispose();
            _flowLmMain = null;

            _flowLmFlow?.Dispose();
            _flowLmFlow = null;

            _sentencePiece?.Dispose();
            _sentencePiece = null;
        }

        #endregion

        #region Prompt

        private class PromptPayload : IBackgroundPayload
        {
            public string Prompt;
            public VoiceInfo Voice;
        }

        public void PromptWithVoice(string prompt, VoiceInfo voice)
        {
            if (string.IsNullOrEmpty(prompt))
                return;

            if (status != ModelStatus.Ready)
                return;

            status = ModelStatus.Generate;
            RunBackground(new PromptPayload { Prompt = prompt, Voice = voice }, RunPrompt);
        }

        [Header("Config")]
        public int MaxFrames = 500;
        public int DiffusionStep = 10;
        public int AudioChunkSize = 12;
        public float Temperature = 0.7f;

        const float EOS_THRESHOLD = -4f;
        const int FRAMES_AFTER_EOS = 3;

        void RunPrompt(PromptPayload payload, CancellationToken cts)
        {
            Dictionary<string, OrtValue> flowState = null;
            Dictionary<string, OrtValue> mimiState = null;
            OrtValue currentLatent = null;
            var stPairs = new List<STPair>();
            System.Random rand = new();

            try
            {
                for (int i = 0; i < DiffusionStep; i++)
                {
                    stPairs.Add(new STPair((float)i / DiffusionStep, (float)(i + 1) / DiffusionStep));
                }

                flowState = InitFlowLmState();
                mimiState = decoder.InitMimiDecoderState();

                using var emptySeq = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 0, 32 });
                using var emptyText = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 0, 1024 });
                using var voiceTensor = OrtValue.CreateTensorValueFromMemory<float>(
                    OrtMemoryInfo.DefaultInstance,
                    payload.Voice.Data,
                    payload.Voice.Shape);

                // ---- Voice conditioning
                using (var res = _flowLmMain.Run(
                    new RunOptions(),
                    TensorUtil.Merge(flowState, ("sequence", emptySeq), ("text_embeddings", voiceTensor)),
                    _flowLmMain.OutputNames))
                {
                    TensorUtil.UpdateState(flowState, res, _flowLmMain.OutputNames);
                }

                List<string> textChunk = TextProcessor.SplitIntoBestSentences(payload.Prompt, _sentencePiece);
                for (int stringIndex = 0; stringIndex < textChunk.Count; stringIndex++)
                {
                    // ---- Text conditioning
                    var ids = EncodeToIds(textChunk[stringIndex]).ToArray();
                    using var tokenTensor = OrtValue.CreateTensorValueFromMemory<long>(
                        OrtMemoryInfo.DefaultInstance, ids, new long[] { 1, ids.Length });

                    using var textRes = _textConditioner.Run(
                        new RunOptions(),
                        new Dictionary<string, OrtValue> { { "token_ids", tokenTensor } },
                        _textConditioner.OutputNames);

                    using var textEmb = TensorUtil.CloneTensor(textRes[0]);

                    using (var res = _flowLmMain.Run(
                        new RunOptions(),
                        TensorUtil.Merge(flowState, ("sequence", emptySeq), ("text_embeddings", textEmb)),
                        _flowLmMain.OutputNames))
                    {
                        TensorUtil.UpdateState(flowState, res, _flowLmMain.OutputNames);
                    }

                    float[] currentLatentData = new float[32];
                    for (int i = 0; i < 32; i++) currentLatentData[i] = float.NaN;

                    int eosStep = -1;
                    var chunk = new List<float[]>();

                    for (int step = 0; step < MaxFrames; step++)
                    {
                        currentLatent = OrtValue.CreateTensorValueFromMemory<float>(
                            OrtMemoryInfo.DefaultInstance, currentLatentData, new long[] { 1, 1, 32 });

                        using var arRes = _flowLmMain.Run(
                            new RunOptions(),
                            TensorUtil.Merge(flowState, ("sequence", currentLatent), ("text_embeddings", emptyText)),
                            _flowLmMain.OutputNames);

                        float eos = arRes[1].GetTensorDataAsSpan<float>()[0];
                        if (eosStep < 0 && eos > EOS_THRESHOLD)
                            eosStep = step;

                        // Flow Matching (LSD)
                        float[] xData = new float[32];
                        double std = Math.Sqrt(Temperature);
                        for (int i = 0; i < 32; i++)
                        {
                            double u1 = 1.0 - rand.NextDouble();
                            double u2 = 1.0 - rand.NextDouble();
                            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                            xData[i] = (float)(randStdNormal * std);
                        }

                        float dt = 1.0f / DiffusionStep;

                        for (int j = 0; j < DiffusionStep; j++)
                        {
                            using var x = OrtValue.CreateTensorValueFromMemory<float>(
                                OrtMemoryInfo.DefaultInstance, xData, new long[] { 1, 32 });

                            using var flowRes = _flowLmFlow.Run(
                                new RunOptions(),
                                new Dictionary<string, OrtValue>
                                {
                                { "c", arRes[0] },
                                { "s", stPairs[j].S },
                                { "t", stPairs[j].T },
                                { "x", x }
                                },
                                _flowLmFlow.OutputNames);

                            var v = flowRes[0].GetTensorDataAsSpan<float>();
                            for (int k = 0; k < 32; k++)
                                xData[k] += v[k] * dt;
                        }

                        float[] finalFrame = (float[])xData.Clone();
                        chunk.Add(finalFrame);

                        Array.Copy(finalFrame, currentLatentData, 32);

                        currentLatent.Dispose();

                        TensorUtil.UpdateState(flowState, arRes, _flowLmMain.OutputNames);

                        if (eosStep >= 0 && step >= eosStep + FRAMES_AFTER_EOS)
                        {
                            float[] audioChunk = decoder.DecodeChunk(chunk, ref mimiState);
                            PostResponse(audioChunk);
                            break;
                        }

                        if (chunk.Count >= AudioChunkSize)
                        {
                            float[] audioChunk = decoder.DecodeChunk(chunk, ref mimiState);
                            PostResponse(audioChunk);
                            chunk.Clear();
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError(e);
            }
            finally
            {
                foreach (var st in stPairs)
                {
                    st.Dispose();
                }
                currentLatent?.Dispose();

                TensorUtil.DisposeState(flowState);
                TensorUtil.DisposeState(mimiState);

                PostStatus(ModelStatus.Ready);
            }
        }

        #endregion

        #region Helpers

        List<long> EncodeToIds(string prompt)
        {
            List<long> tokenIds = new List<long>();
            try
            {
                List<string> tokens = _sentencePiece.EncodeToPieces(prompt);
                for (int i = 0; i < tokens.Count; i++)
                    tokenIds.Add(_sentencePiece.PieceToId(tokens[i]));
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"An unexpected error occurred: {ex.Message}");
            }
            return tokenIds;
        }

        public Dictionary<string, OrtValue> InitFlowLmState()
        {
            var state = new Dictionary<string, OrtValue>();
            long[] kvShape = { 2, 1, 1000, 16, 64 };

            for (int i = 0; i <= 15; i += 3)
            {
                state[$"state_{i}"] = TensorUtil.CreateEmptyFloatTensor(kvShape);
                state[$"state_{i + 1}"] = TensorUtil.CreateEmptyFloatTensor(new long[] { 0 });
                state[$"state_{i + 2}"] = TensorUtil.CreateInt64Tensor(0);
            }
            return state;
        }

        #endregion

        public delegate void ResponseGeneratedDelegate(float[] response);
        public event ResponseGeneratedDelegate OnResponseGenerated;

        void PostResponse(float[] response)
        {
            unityContext?.Post(_ => OnResponseGenerated?.Invoke(response), null);
        }

    }

    public class STPair : IDisposable
    {
        public OrtValue S { get; }
        public OrtValue T { get; }

        public STPair(float s, float t)
        {
            S = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance, new[] { s }, new long[] { 1, 1 });
            T = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance, new[] { t }, new long[] { 1, 1 });
        }

        public void Dispose()
        {
            S.Dispose();
            T.Dispose();
        }
    }
}
