using System;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace PocketTTS
{
    public static class NpyUtil
    {
        public static (float[] data, long[] shape) ParseNpyFloat32(byte[] bytes)
        {
            if (bytes[0] != 0x93 || bytes[1] != 'N' || bytes[2] != 'U' || bytes[3] != 'M' || bytes[4] != 'P' || bytes[5] != 'Y')
                throw new Exception("Invalid NPY magic");

            int major = bytes[6];
            int minor = bytes[7];

            int headerLen;
            int offset;
            if (major == 1)
            {
                headerLen = BitConverter.ToUInt16(bytes, 8);
                offset = 10;
            }
            else if (major == 2 || major == 3)
            {
                headerLen = (int)BitConverter.ToUInt32(bytes, 8);
                offset = 12;
            }
            else
            {
                throw new Exception($"Unsupported NPY version {major}.{minor}");
            }

            string header = Encoding.ASCII.GetString(bytes, offset, headerLen);
            
            if (!header.Contains("'descr': '<f4'"))
                throw new Exception("Unsupported NPY dtype, only <f4 (float32) is supported");

            var shapeMatch = Regex.Match(header, @"'shape':\s*\(([^)]*)\)");
            if (!shapeMatch.Success)
                throw new Exception("Could not parse shape from NPY header");

            long[] shape = shapeMatch.Groups[1].Value
                .Split(',')
                .Select(s => s.Trim())
                .Where(s => s.Length > 0)
                .Select(long.Parse)
                .ToArray();

            int dataOffset = offset + headerLen;
            int dataBytes = bytes.Length - dataOffset;
            float[] data = new float[dataBytes / 4];
            Buffer.BlockCopy(bytes, dataOffset, data, 0, dataBytes);

            return (data, shape);
        }
    }
}
