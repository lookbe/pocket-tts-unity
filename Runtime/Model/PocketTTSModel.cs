using Microsoft.ML.OnnxRuntime;
using PocketTts;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace PocketTTS
{
    public class PocketTTSModel : BackgroundRunner
    {
        [Header("Models")]
        public string tokenizerPath = string.Empty;
        public string textConditionerPath = string.Empty;
        public string flowLmMainPath = string.Empty;
        public string flowLmFlowPath = string.Empty;
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

        private InferenceSession _textConditioner;
        private InferenceSession _flowLmMain;
        private InferenceSession _flowLmFlow;
        private InferenceSession _mimiDecoder;
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

                var opt = TensorUtil.GetMobileSessionOptions();

                _textConditioner = new InferenceSession(textConditionerPath, opt);
                _flowLmMain = new InferenceSession(flowLmMainPath, opt);
                _flowLmFlow = new InferenceSession(flowLmFlowPath, opt);
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
            _textConditioner?.Dispose();
            _textConditioner = null;

            _flowLmMain?.Dispose();
            _flowLmMain = null;

            _flowLmFlow?.Dispose();
            _flowLmFlow = null;

            _mimiDecoder?.Dispose();
            _mimiDecoder = null;

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

        private struct SynthesisItem
        {
            public float[] Conditioning;
            public float[] Noise;
            public bool IsFinal;
        }

        void RunPrompt(PromptPayload payload, CancellationToken cts)
        {
            Dictionary<string, OrtValue> flowState = null;
            Dictionary<string, OrtValue> mimiState = null;
            var stPairs = new List<STPair>();
            System.Random rand = new();

            // Pipelining: Synthesis Queue, Potential Buffer Pools, and Feedback
            var synthQueue = new BlockingCollection<SynthesisItem>(new ConcurrentQueue<SynthesisItem>());
            var latentFeedbackQueue = new BlockingCollection<float[]>(new ConcurrentQueue<float[]>());
            var bufferPool = new ConcurrentStack<float[]>();
            var noisePool = new ConcurrentStack<float[]>();
            
            Func<int, float[]> getCondBuffer = (size) => bufferPool.TryPop(out var b) && b.Length >= size ? b : new float[size];
            Func<float[]> getNoiseBuffer = () => noisePool.TryPop(out var b) ? b : new float[32];
            
            Action<float[]> returnCondBuffer = (b) => bufferPool.Push(b);
            Action<float[]> returnNoiseBuffer = (b) => noisePool.Push(b);

            // Set high priority for this thread
            Thread.CurrentThread.Priority = System.Threading.ThreadPriority.AboveNormal;

            // Pre-allocate buffers for inference
            float[] currentLatentData = new float[32];
            float[] xData = new float[32];
            long[] currentLatentShape = { 1, 1, 32 };
            long[] xShape = { 1, 32 };
            
            // Adaptive conditioning size
            int condDim = 32; 

            // Pre-generate Gaussian noise for the entire sequence to avoid Box-Muller in hot loop
            int noiseCount = MaxFrames * 32; // 500 * 32
            float[] noiseBuffer = new float[noiseCount];
            for (int n = 0; n < noiseCount; n += 2)
            {
                double u1 = 1.0 - rand.NextDouble();
                double u2 = 1.0 - rand.NextDouble();
                double radius = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;
                noiseBuffer[n] = (float)(radius * Math.Cos(theta));
                if (n + 1 < noiseCount)
                    noiseBuffer[n + 1] = (float)(radius * Math.Sin(theta));
            }
            float noiseStd = (float)Math.Sqrt(Temperature);
            for (int n = 0; n < noiseCount; n++) noiseBuffer[n] *= noiseStd;

            // Persistent dictionaries for inputs to avoid per-step allocations
            Dictionary<string, OrtValue> mainInputs = null;
            Dictionary<string, OrtValue> flowInputs = null;

            // Pin buffers to create persistent OrtValue tensors
            GCHandle latentHandle = GCHandle.Alloc(currentLatentData, GCHandleType.Pinned);
            GCHandle xHandle = GCHandle.Alloc(xData, GCHandleType.Pinned);

            OrtValue currentLatent = null;
            OrtValue xTensor = null;

            try
            {
                currentLatent = OrtValue.CreateTensorValueFromMemory<float>(
                    OrtMemoryInfo.DefaultInstance, currentLatentData, currentLatentShape);
                xTensor = OrtValue.CreateTensorValueFromMemory<float>(
                    OrtMemoryInfo.DefaultInstance, xData, xShape);

                for (int i = 0; i < DiffusionStep; i++)
                {
                    stPairs.Add(new STPair((float)i / DiffusionStep, (float)(i + 1) / DiffusionStep));
                }

                flowState = InitFlowLmState();
                mimiState = InitMimiDecoderState();

                using var emptySeq = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 0, 32 });
                using var emptyText = TensorUtil.CreateEmptyFloatTensor(new long[] { 1, 0, 1024 });
                using var voiceTensor = OrtValue.CreateTensorValueFromMemory<float>(
                    OrtMemoryInfo.DefaultInstance, payload.Voice.Data, payload.Voice.Shape);

                mainInputs = new Dictionary<string, OrtValue>(flowState);
                mainInputs["sequence"] = emptySeq;
                mainInputs["text_embeddings"] = voiceTensor;

                // ---- Voice conditioning
                using (var res = _flowLmMain.Run(new RunOptions(), mainInputs, _flowLmMain.OutputNames))
                {
                    TensorUtil.UpdateState(flowState, res, _flowLmMain.OutputNames);
                }

                List<string> textChunk = TextProcessor.SplitIntoBestSentences(payload.Prompt, _sentencePiece);
                for (int stringIndex = 0; stringIndex < textChunk.Count; stringIndex++)
                {
                    if (cts.IsCancellationRequested) break;

                    var ids = EncodeToIds(textChunk[stringIndex]).ToArray();
                    using var tokenTensor = OrtValue.CreateTensorValueFromMemory<long>(
                        OrtMemoryInfo.DefaultInstance, ids, new long[] { 1, ids.Length });

                    using var textRes = _textConditioner.Run(
                        new RunOptions(),
                        new Dictionary<string, OrtValue> { { "token_ids", tokenTensor } },
                        _textConditioner.OutputNames);

                    // Update mainInputs for text conditioning
                    foreach (var kv in flowState) mainInputs[kv.Key] = kv.Value;
                    mainInputs["sequence"] = emptySeq;
                    mainInputs["text_embeddings"] = textRes[0];

                    using (var res = _flowLmMain.Run(new RunOptions(), mainInputs, _flowLmMain.OutputNames))
                    {
                        TensorUtil.UpdateState(flowState, res, _flowLmMain.OutputNames);
                    }

                    for (int i = 0; i < 32; i++) currentLatentData[i] = float.NaN;

                    int eosStep = -1;

                    flowInputs = new Dictionary<string, OrtValue>();

                    // Initialize mainInputs once with persistent dictionary structure
                    // Since UpdateState uses CloneInto, the dictionary references stay valid.
                    // Stage 2: Synthesis Worker (Pipelined)
                    var synthTask = Task.Run(() =>
                    {
                        var workerFlowInputs = new Dictionary<string, OrtValue>();
                        float[] workerXData = new float[32];
                        long[] workerXShape = { 1, 32 };
                        GCHandle workerXHandle = GCHandle.Alloc(workerXData, GCHandleType.Pinned);
                        
                        using var workerXTensor = OrtValue.CreateTensorValueFromMemory<float>(
                            OrtMemoryInfo.DefaultInstance, workerXData, workerXShape);

                        var workerChunk = new List<float[]>();

                        // Set worker to highest priority to avoid hiccups
                        Thread.CurrentThread.Priority = System.Threading.ThreadPriority.Highest;

                        try
                        {
                            foreach (var item in synthQueue.GetConsumingEnumerable(cts))
                            {
                                if (item.IsFinal) break;

                                // 1. Flow Matching
                                Array.Copy(item.Noise, workerXData, 32);
                                float dt = 1.0f / DiffusionStep;

                                using var condTensor = OrtValue.CreateTensorValueFromMemory<float>(
                                    OrtMemoryInfo.DefaultInstance, item.Conditioning, new long[] { 1, item.Conditioning.Length });

                                workerFlowInputs["c"] = condTensor;
                                for (int j = 0; j < DiffusionStep; j++)
                                {
                                    workerFlowInputs["s"] = stPairs[j].S;
                                    workerFlowInputs["t"] = stPairs[j].T;
                                    workerFlowInputs["x"] = workerXTensor;

                                    using var flowRes = _flowLmFlow.Run(new RunOptions(), workerFlowInputs, _flowLmFlow.OutputNames);
                                    var v = flowRes[0].GetTensorDataAsSpan<float>();

                                    int k = 0;
                                    if (Vector.IsHardwareAccelerated && (32 % Vector<float>.Count == 0))
                                    {
                                        int vectorSize = Vector<float>.Count;
                                        Vector<float> dtVec = new Vector<float>(dt);
                                        var vVectors = MemoryMarshal.Cast<float, Vector<float>>(v);
                                        var xVectors = MemoryMarshal.Cast<float, Vector<float>>(workerXData.AsSpan());
                                        for (int vIdx = 0; vIdx < vVectors.Length; vIdx++)
                                            xVectors[vIdx] = xVectors[vIdx] + (vVectors[vIdx] * dtVec);
                                        k = vVectors.Length * vectorSize;
                                    }
                                    for (; k < 32; k++)
                                        workerXData[k] += v[k] * dt;
                                }

                                // Pass REFINED latent back to producer for next AR step
                                var refinedLatent = new float[32];
                                Array.Copy(workerXData, refinedLatent, 32);
                                latentFeedbackQueue.Add(refinedLatent);

                                // 2. Decoding
                                workerChunk.Add(refinedLatent);

                                if (workerChunk.Count >= AudioChunkSize)
                                {
                                    float[] audioChunk = DecodeChunk(workerChunk, ref mimiState, _mimiDecoder);
                                    PostResponse(audioChunk);
                                    workerChunk.Clear();
                                }

                                // Return buffers to pool
                                returnCondBuffer(item.Conditioning);
                                returnNoiseBuffer(item.Noise);
                            }

                            if (workerChunk.Count > 0)
                            {
                                float[] audioChunk = DecodeChunk(workerChunk, ref mimiState, _mimiDecoder);
                                PostResponse(audioChunk);
                            }
                        }
                        finally
                        {
                            workerXHandle.Free();
                        }
                    }, cts);

                    // Stage 1: AR Generator (Producer)
                    mainInputs = new Dictionary<string, OrtValue>(flowState);
                    mainInputs["sequence"] = currentLatent;
                    mainInputs["text_embeddings"] = emptyText;

                    for (int step = 0; step < MaxFrames; step++)
                    {
                        if (cts.IsCancellationRequested) break;

                        using var arRes = _flowLmMain.Run(new RunOptions(), mainInputs, _flowLmMain.OutputNames);

                        var arOutputSpan = arRes[0].GetTensorDataAsSpan<float>();
                        if (step == 0) condDim = arOutputSpan.Length;

                        float eos = arRes[1].GetTensorDataAsSpan<float>()[0];
                        if (eosStep < 0 && eos > EOS_THRESHOLD)
                            eosStep = step;

                        // Package conditioning and noise, then ship to worker
                        var synthItem = new SynthesisItem 
                        { 
                            Conditioning = getCondBuffer(condDim), 
                            Noise = getNoiseBuffer(), 
                            IsFinal = false 
                        };
                        
                        arOutputSpan.CopyTo(synthItem.Conditioning);
                        
                        int noiseIdx = step * 32;
                        Array.Copy(noiseBuffer, noiseIdx, synthItem.Noise, 0, 32);

                        synthQueue.Add(synthItem);

                        // WAIT for refined latent before next AR step
                        if (latentFeedbackQueue.TryTake(out var refined, 2000, cts))
                        {
                            Array.Copy(refined, currentLatentData, 32);
                        }

                        if (eosStep >= 0 && step >= eosStep + FRAMES_AFTER_EOS)
                            break;

                        TensorUtil.UpdateState(flowState, arRes, _flowLmMain.OutputNames);
                    }

                    synthQueue.Add(new SynthesisItem { IsFinal = true });
                    synthTask.Wait(cts);
                }
            }
            catch (Exception e)
            {
                Debug.LogError(e);
            }
            finally
            {
                foreach (var st in stPairs) st.Dispose();
                stPairs.Clear();
                currentLatent?.Dispose();
                xTensor?.Dispose();

                if (latentHandle.IsAllocated) latentHandle.Free();
                if (xHandle.IsAllocated) xHandle.Free();

                FreeDecoderResources();

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

        private float[] _decoderFlattenBuffer;
        private GCHandle _decoderFlattenHandle;
        private OrtValue _decoderLatentTensor;
        private Dictionary<string, OrtValue> _decoderInputs;

        private float[] DecodeChunk(List<float[]> chunk, ref Dictionary<string, OrtValue> state, InferenceSession decoder)
        {
            try
            {
                int count = chunk.Count;
                int size = count * 32;

                if (_decoderFlattenBuffer == null || _decoderFlattenBuffer.Length < size)
                {
                    if (_decoderFlattenHandle.IsAllocated) _decoderFlattenHandle.Free();
                    _decoderLatentTensor?.Dispose();

                    _decoderFlattenBuffer = new float[size * 2];
                    _decoderFlattenHandle = GCHandle.Alloc(_decoderFlattenBuffer, GCHandleType.Pinned);
                    _decoderLatentTensor = null;
                }

                if (_decoderLatentTensor == null || _decoderLatentTensor.GetTensorTypeAndShape().Shape[1] != count)
                {
                    _decoderLatentTensor?.Dispose();
                    _decoderLatentTensor = OrtValue.CreateTensorValueFromMemory<float>(
                        OrtMemoryInfo.DefaultInstance, _decoderFlattenBuffer, new long[] { 1, count, 32 });
                }

                for (int i = 0; i < count; i++)
                {
                    Array.Copy(chunk[i], 0, _decoderFlattenBuffer, i * 32, 32);
                }

                if (_decoderInputs == null)
                {
                    _decoderInputs = new Dictionary<string, OrtValue>(state);
                }
                else
                {
                    foreach (var kv in state) _decoderInputs[kv.Key] = kv.Value;
                }
                _decoderInputs["latent"] = _decoderLatentTensor;

                using var res = decoder.Run(new RunOptions(), _decoderInputs, decoder.OutputNames);
                var audioOutput = res[0];
                ReadOnlySpan<float> audioSpan = audioOutput.GetTensorDataAsSpan<float>();
                TensorUtil.UpdateState(state, res, decoder.OutputNames);
                return audioSpan.ToArray();
            }
            catch (Exception e)
            {
                Debug.LogError(e);
            }
            return new float[0];
        }

        private void FreeDecoderResources()
        {
            if (_decoderFlattenHandle.IsAllocated) _decoderFlattenHandle.Free();
            _decoderLatentTensor?.Dispose();
            _decoderLatentTensor = null;
            _decoderFlattenBuffer = null;
            _decoderInputs = null;
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
