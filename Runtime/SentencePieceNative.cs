using System.Runtime.InteropServices;
using System;

public static class SentencePieceNative
{
    private const string LibName = "SentencePieceWrapper"; 

    [StructLayout(LayoutKind.Sequential)]
    public struct StringArray
    {
        public IntPtr pieces; 
        public int count;
    }

    // Processor Management
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr spw_load_processor(string modelPath);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void spw_dispose_processor(IntPtr handle);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern StringArray spw_encode_to_pieces(IntPtr handle, string text);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void spw_free_string_array(StringArray array);
    
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void spw_free_string(IntPtr str);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int spw_piece_to_id(IntPtr handle, string piece);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr spw_id_to_piece(IntPtr handle, int id);
}