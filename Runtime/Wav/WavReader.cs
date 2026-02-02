using System;
using System.IO;
using System.Text;

namespace PocketTTS
{
    public static class WavReader
    {
        public static float[] LoadWav(string filePath, int targetSampleRate = 24000)
        {
            using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            using (var br = new BinaryReader(fs))
            {
                // --- 1. MINIMAL WAV HEADER PARSING ---
                br.ReadBytes(22); // Skip RIFF header and parts of fmt chunk
                short channels = br.ReadInt16();
                int sourceSampleRate = br.ReadInt32();
                br.ReadBytes(6); // Skip byte rate and block align
                short bitDepth = br.ReadInt16();

                // Find 'data' chunk
                while (new string(br.ReadChars(4)) != "data")
                {
                    int chunkSize = br.ReadInt32();
                    br.ReadBytes(chunkSize);
                }

                int dataSize = br.ReadInt32();
                int totalSamples = dataSize / (bitDepth / 8);

                // --- 2. CONVERT TO FLOAT PCM ---
                float[] pcmData = new float[totalSamples];
                for (int i = 0; i < totalSamples; i++)
                {
                    if (bitDepth == 16)
                        pcmData[i] = br.ReadInt16() / 32768f;
                    else if (bitDepth == 32)
                        pcmData[i] = br.ReadSingle();
                }

                // --- 3. CHANNEL CHECK & MONO MIXING ---
                float[] monoSamples;
                if (channels == 1)
                {
                    monoSamples = pcmData;
                }
                else
                {
                    int frames = totalSamples / channels;
                    monoSamples = new float[frames];
                    for (int i = 0; i < frames; i++)
                    {
                        float sum = 0;
                        for (int c = 0; c < channels; c++)
                        {
                            sum += pcmData[i * channels + c];
                        }
                        monoSamples[i] = sum / channels;
                    }
                }

                // --- 4. RESAMPLE USING WDL ---
                if (sourceSampleRate == targetSampleRate) return monoSamples;

                return ApplyWdlResample(monoSamples, sourceSampleRate, targetSampleRate);
            }
        }

        private static float[] ApplyWdlResample(float[] input, int srcRate, int dstRate)
        {
            var resampler = new WdlResampler();
            resampler.SetMode(true, 6, false);
            resampler.SetRates(srcRate, dstRate);

            int channels = 1; // Already mixed to mono
            float[] inBuffer;
            int inBufferOffset;

            int inNeeded = resampler.ResamplePrepare(input.Length, channels, out inBuffer, out inBufferOffset);
            Array.Copy(input, 0, inBuffer, inBufferOffset, Math.Min(input.Length, inNeeded));

            double ratio = (double)dstRate / srcRate;
            int outCapacity = (int)(input.Length * ratio) + 100;
            float[] outBuffer = new float[outCapacity];

            int framesOut = resampler.ResampleOut(outBuffer, 0, inNeeded, outCapacity, channels);

            float[] finalOutput = new float[framesOut];
            Array.Copy(outBuffer, finalOutput, framesOut);
            return finalOutput;
        }

    }
}
