using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public class SentencePieceWrapper : IDisposable
{
    // Opaque handle to the native C++ SentencePieceProcessor object
    private IntPtr _processorHandle; 

    /// <summary>
    /// Loads the SentencePiece model from the specified path.
    /// </summary>
    /// <param name="modelPath">The path to the SentencePiece model file (.model).</param>
    public SentencePieceWrapper(string modelPath)
    {
        _processorHandle = SentencePieceNative.spw_load_processor(modelPath);
        if (_processorHandle == IntPtr.Zero)
        {
            throw new InvalidOperationException($"Failed to load SentencePiece model from {modelPath}. Ensure the path is correct and the native library is available.");
        }
    }

    /// <summary>
    /// Encodes a text string into a list of SentencePiece token pieces.
    /// </summary>
    /// <param name="text">The input text to tokenize.</param>
    /// <returns>A list of token pieces (strings).</returns>
    public List<string> EncodeToPieces(string text)
    {
        if (_processorHandle == IntPtr.Zero)
        {
            throw new ObjectDisposedException(nameof(SentencePieceWrapper), "The native library handle is invalid or disposed.");
        }

        var pieces = new List<string>();
        SentencePieceNative.StringArray array = SentencePieceNative.spw_encode_to_pieces(_processorHandle, text);

        try
        {
            IntPtr currentPtr = array.pieces;
            int pointerSize = IntPtr.Size;

            for (int i = 0; i < array.count; i++)
            {
                // Read the pointer to the individual C-string in the array
                IntPtr piecePtr = Marshal.ReadIntPtr(currentPtr, i * pointerSize);
                if (piecePtr != IntPtr.Zero)
                {
                    // Convert C-string (UTF-8) to C# string
                    pieces.Add(Marshal.PtrToStringUTF8(piecePtr) ?? string.Empty);

                    // Crucially, free the individual string piece allocated by C++
                    SentencePieceNative.spw_free_string(piecePtr); 
                }
            }
        }
        finally
        {
            // Free the array of pointers itself allocated by C++
            SentencePieceNative.spw_free_string_array(array);
        }
        return pieces;
    }

    /// <summary>
    /// Converts a single SentencePiece piece string to its corresponding ID.
    /// </summary>
    public int PieceToId(string piece)
    {
        if (_processorHandle == IntPtr.Zero) return -1;
        return SentencePieceNative.spw_piece_to_id(_processorHandle, piece);
    }

    /// <summary>
    /// Converts a single ID to its corresponding SentencePiece piece string.
    /// </summary>
    public string IdToPiece(int id)
    {
        if (_processorHandle == IntPtr.Zero) return string.Empty;

        // Call native function
        IntPtr piecePtr = SentencePieceNative.spw_id_to_piece(_processorHandle, id);
        
        // Marshal and convert
        string piece = Marshal.PtrToStringUTF8(piecePtr) ?? string.Empty;

        // Free the memory allocated by the C++ function
        SentencePieceNative.spw_free_string(piecePtr);
        
        return piece;
    }
    
    // --- IDisposable Implementation ---

    public void Dispose()
    {
        if (_processorHandle != IntPtr.Zero)
        {
            SentencePieceNative.spw_dispose_processor(_processorHandle);
            _processorHandle = IntPtr.Zero;
        }
        GC.SuppressFinalize(this);
    }

    ~SentencePieceWrapper()
    {
        Dispose();
    }
}