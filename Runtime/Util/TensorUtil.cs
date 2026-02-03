using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace PocketTTS
{
    public static class TensorUtil
    {
        public static void CloneInto(OrtValue src, OrtValue dest)
        {
            OrtTensorTypeAndShapeInfo srcInfo = src.GetTensorTypeAndShape();
            OrtTensorTypeAndShapeInfo destInfo = dest.GetTensorTypeAndShape();

            if (!srcInfo.Shape.SequenceEqual(destInfo.Shape))
                throw new ArgumentException("Shapes must match for CloneInto");
            
            if (srcInfo.ElementDataType != destInfo.ElementDataType)
                throw new ArgumentException("Data types must match for CloneInto");

            switch (srcInfo.ElementDataType)
            {
                case TensorElementType.Float:
                    {
                        var s = src.GetTensorDataAsSpan<float>();
                        var d = dest.GetTensorDataAsSpan<float>();
                        var target = MemoryMarshal.CreateSpan(ref MemoryMarshal.GetReference(d), d.Length);
                        s.CopyTo(target);
                        break;
                    }
                case TensorElementType.Int64:
                    {
                        var s = src.GetTensorDataAsSpan<long>();
                        var d = dest.GetTensorDataAsSpan<long>();
                        var target = MemoryMarshal.CreateSpan(ref MemoryMarshal.GetReference(d), d.Length);
                        s.CopyTo(target);
                        break;
                    }
                case TensorElementType.Bool:
                    {
                        var s = src.GetTensorDataAsSpan<bool>();
                        var d = dest.GetTensorDataAsSpan<bool>();
                        var target = MemoryMarshal.CreateSpan(ref MemoryMarshal.GetReference(d), d.Length);
                        s.CopyTo(target);
                        break;
                    }
                default:
                    throw new NotSupportedException($"Unsupported tensor type {srcInfo.ElementDataType}");
            }
        }

        public static OrtValue CloneTensor(OrtValue src)
        {
            OrtTensorTypeAndShapeInfo info = src.GetTensorTypeAndShape();
            long[] shape = info.Shape;
            TensorElementType dtype = info.ElementDataType;

            switch (dtype)
            {
                case TensorElementType.Float:
                    {
                        var data = src.GetTensorDataAsSpan<float>().ToArray();
                        return OrtValue.CreateTensorValueFromMemory<float>(
                            OrtMemoryInfo.DefaultInstance, data, shape);
                    }
                case TensorElementType.Int64:
                    {
                        var data = src.GetTensorDataAsSpan<long>().ToArray();
                        return OrtValue.CreateTensorValueFromMemory<long>(
                            OrtMemoryInfo.DefaultInstance, data, shape);
                    }
                case TensorElementType.Bool:
                    {
                        var data = src.GetTensorDataAsSpan<bool>().ToArray();
                        return OrtValue.CreateTensorValueFromMemory<bool>(
                            OrtMemoryInfo.DefaultInstance, data, shape);
                    }
                default:
                    throw new NotSupportedException($"Unsupported tensor type {dtype}");
            }
        }

        public static Dictionary<string, OrtValue> Merge(Dictionary<string, OrtValue> baseDict, params (string, OrtValue)[] extra)
        {
            var d = new Dictionary<string, OrtValue>(baseDict);
            foreach (var e in extra)
            {
                d[e.Item1] = e.Item2;
            }
            return d;
        }

        public static void UpdateState(Dictionary<string, OrtValue> state, IReadOnlyList<OrtValue> res, IReadOnlyList<string> outputNames)
        {
            for (int i = 0; i < outputNames.Count; i++)
            {
                string outputName = outputNames[i];

                if (outputName.StartsWith("out_state_"))
                {
                    string stateIdx = outputName.Replace("out_state_", "");
                    string key = $"state_{stateIdx}";

                    if (state.TryGetValue(key, out var existing) && existing != null)
                    {
                        CloneInto(res[i], existing);
                    }
                    else
                    {
                        state[key]?.Dispose();
                        state[key] = TensorUtil.CloneTensor(res[i]);
                    }
                }
            }
        }

        public static void DisposeState(Dictionary<string, OrtValue> state)
        {
            if (state == null) return;
            foreach (var v in state.Values) v.Dispose();
            state.Clear();
        }

        public static OrtValue CreateBoolTensor(bool value)
        {
            return OrtValue.CreateTensorValueFromMemory<bool>(OrtMemoryInfo.DefaultInstance, new bool[] { value }, new long[] { 1 });
        }

        public static OrtValue CreateEmptyFloatTensor(long[] shape)
        {
            long elementCount = shape.Aggregate(1L, (a, b) => b == 0 ? 0 : a * b);
            float[] buffer = new float[elementCount];
            return OrtValue.CreateTensorValueFromMemory<float>(OrtMemoryInfo.DefaultInstance, buffer, shape);
        }

        public static OrtValue CreateInt64Tensor(long value)
        {
            return OrtValue.CreateTensorValueFromMemory<long>(OrtMemoryInfo.DefaultInstance, new long[] { value }, new long[] { 1 });
        }

        public static OrtValue CreateInt64Tensor(long[] values, long[] shape)
        {
            return OrtValue.CreateTensorValueFromMemory<long>(OrtMemoryInfo.DefaultInstance, values, shape);
        }

        public static SessionOptions GetMobileSessionOptions()
        {
            var opt = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                IntraOpNumThreads = 2, // Balance for mobile
                InterOpNumThreads = 1
            };
            return opt;
        }

        public static void UpdateInput(Dictionary<string, OrtValue> inputs, string name, OrtValue value)
        {
            inputs[name] = value;
        }

        public static void UpdateInputs(Dictionary<string, OrtValue> inputs, params (string, OrtValue)[] extra)
        {
            foreach (var e in extra)
            {
                inputs[e.Item1] = e.Item2;
            }
        }
    }
}
