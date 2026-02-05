using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Text;
using System.Linq;
using System.Globalization;

namespace PocketTts
{
    public class TextProcessor
    {
        private static readonly Dictionary<char, string> UnicodeMap = new Dictionary<char, string>
        {
            {'à', "a"}, {'á', "a"}, {'â', "a"}, {'ã', "a"}, {'ä', "a"}, {'å', "a"}, {'æ', "ae"}, {'ç', "c"},
            {'è', "e"}, {'é', "e"}, {'ê', "e"}, {'ë', "e"}, {'ì', "i"}, {'í', "i"}, {'î', "i"}, {'ï', "i"},
            {'ñ', "n"}, {'ò', "o"}, {'ó', "o"}, {'ô', "o"}, {'õ', "o"}, {'ö', "o"}, {'ø', "o"}, {'ù', "u"},
            {'ú', "u"}, {'û', "u"}, {'ü', "u"}, {'ý', "y"}, {'ÿ', "y"}, {'ß', "ss"}, {'œ', "oe"}, {'ð', "d"},
            {'þ', "th"}, {'À', "A"}, {'Á', "A"}, {'Â', "A"}, {'Ã', "A"}, {'Ä', "A"}, {'Å', "A"}, {'Æ', "AE"},
            {'Ç', "C"}, {'È', "E"}, {'É', "E"}, {'Ê', "E"}, {'Ë', "E"}, {'Ì', "I"}, {'Í', "I"}, {'Î', "I"},
            {'Ï', "I"}, {'Ñ', "N"}, {'Ò', "O"}, {'Ó', "O"}, {'Ô', "O"}, {'Õ', "O"}, {'Ö', "O"}, {'Ø', "O"},
            {'Ù', "U"}, {'Ú', "U"}, {'Û', "U"}, {'Ü', "U"}, {'Ý', "Y"}, {'\u201C', "\""}, {'\u201D', "\""},
            {'\u2018', "'"}, {'\u2019', "'"}, {'\u2026', "..."}, {'\u2013', "-"}, {'\u2014', "-"}
        };

        private static readonly string[] Ones = { "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
        private static readonly string[] Tens = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
        private static readonly string[] OrdinalOnes = { "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth" };
        private static readonly string[] OrdinalTens = { "", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth" };

        private static readonly (Regex Regex, string Replacement)[] Abbreviations = {
            (new Regex(@"\bmrs\.", RegexOptions.IgnoreCase), "misuss"),
            (new Regex(@"\bms\.", RegexOptions.IgnoreCase), "miss"),
            (new Regex(@"\bmr\.", RegexOptions.IgnoreCase), "mister"),
            (new Regex(@"\bdr\.", RegexOptions.IgnoreCase), "doctor"),
            (new Regex(@"\bst\.", RegexOptions.IgnoreCase), "saint"),
            (new Regex(@"\bco\.", RegexOptions.IgnoreCase), "company"),
            (new Regex(@"\bjr\.", RegexOptions.IgnoreCase), "junior"),
            (new Regex(@"\bmaj\.", RegexOptions.IgnoreCase), "major"),
            (new Regex(@"\bgen\.", RegexOptions.IgnoreCase), "general"),
            (new Regex(@"\bdrs\.", RegexOptions.IgnoreCase), "doctors"),
            (new Regex(@"\brev\.", RegexOptions.IgnoreCase), "reverend"),
            (new Regex(@"\blt\.", RegexOptions.IgnoreCase), "lieutenant"),
            (new Regex(@"\bhon\.", RegexOptions.IgnoreCase), "honorable"),
            (new Regex(@"\bsgt\.", RegexOptions.IgnoreCase), "sergeant"),
            (new Regex(@"\bcapt\.", RegexOptions.IgnoreCase), "captain"),
            (new Regex(@"\besq\.", RegexOptions.IgnoreCase), "esquire"),
            (new Regex(@"\bltd\.", RegexOptions.IgnoreCase), "limited"),
            (new Regex(@"\bcol\.", RegexOptions.IgnoreCase), "colonel"),
            (new Regex(@"\bft\.", RegexOptions.IgnoreCase), "fort")
        };

        private static readonly (Regex Regex, string Replacement)[] CasedAbbreviations = {
            (new Regex(@"\bTTS\b"), "text to speech"),
            (new Regex(@"\bHz\b"), "hertz"),
            (new Regex(@"\bkHz\b"), "kilohertz"),
            (new Regex(@"\bKBs\b"), "kilobytes"),
            (new Regex(@"\bKB\b"), "kilobyte"),
            (new Regex(@"\bMBs\b"), "megabytes"),
            (new Regex(@"\bMB\b"), "megabyte"),
            (new Regex(@"\bGBs\b"), "gigabytes"),
            (new Regex(@"\bGB\b"), "gigabyte"),
            (new Regex(@"\bTBs\b"), "terabytes"),
            (new Regex(@"\bTB\b"), "terabyte"),
            (new Regex(@"\bAPIs\b"), "a p i's"),
            (new Regex(@"\bAPI\b"), "a p i"),
            (new Regex(@"\bCLIs\b"), "c l i's"),
            (new Regex(@"\bCLI\b"), "c l i"),
            (new Regex(@"\bCPUs\b"), "c p u's"),
            (new Regex(@"\bCPU\b"), "c p u"),
            (new Regex(@"\bGPUs\b"), "g p u's"),
            (new Regex(@"\bGPU\b"), "g p u"),
            (new Regex(@"\bAve\b"), "avenue"),
            (new Regex(@"\betc\b"), "etcetera")
        };

        private static readonly (Regex Regex, string Replacement)[] SpecialCharacters = {
            (new Regex(@"@"), " at "), (new Regex(@"&"), " and "), (new Regex(@"%"), " percent "),
            (new Regex(@":"), "."), (new Regex(@";"), ","), (new Regex(@"\+"), " plus "),
            (new Regex(@"\\"), " backslash "), (new Regex(@"~"), " about "),
            (new Regex(@"(^| )<3"), " heart "), (new Regex(@"<="), " less than or equal to "),
            (new Regex(@">="), " greater than or equal to "), (new Regex(@"<"), " less than "),
            (new Regex(@">"), " greater than "), (new Regex(@"="), " equals "),
            (new Regex(@"\/"), " slash "), (new Regex(@"_"), " ")
        };

        public static string NumberToWords(long num, string andword = "", string zero = "zero", int group = 0)
        {
            if (num == 0) return zero;
            if (num < 0) return "minus " + NumberToWords(Math.Abs(num), andword, zero, group);

            if (group == 2 && num > 1000 && num < 10000)
            {
                long high = num / 100;
                long low = num % 100;
                if (low == 0) return NumberToWords(high) + " hundred";
                else if (low < 10) return NumberToWords(high) + " " + (zero == "oh" ? "oh" : zero) + " " + Ones[low];
                else return NumberToWords(high) + " " + NumberToWords(low);
            }

            return ConvertInternal(num, andword);
        }

        private static string ConvertInternal(long n, string andword)
        {
            if (n < 20) return Ones[n];
            if (n < 100) return Tens[n / 10] + (n % 10 != 0 ? " " + Ones[n % 10] : "");
            if (n < 1000)
            {
                long remainder = n % 100;
                return Ones[n / 100] + " hundred" + (remainder != 0 ? (string.IsNullOrEmpty(andword) ? " " : " " + andword + " ") + ConvertInternal(remainder, andword) : "");
            }
            if (n < 1000000)
            {
                long thousands = n / 1000;
                long remainder = n % 1000;
                return ConvertInternal(thousands, andword) + " thousand" + (remainder != 0 ? " " + ConvertInternal(remainder, andword) : "");
            }
            if (n < 1000000000)
            {
                long millions = n / 1000000;
                long remainder = n % 1000000;
                return ConvertInternal(millions, andword) + " million" + (remainder != 0 ? " " + ConvertInternal(remainder, andword) : "");
            }
            long billions = n / 1000000000;
            long remBillions = n % 1000000000;
            return ConvertInternal(billions, andword) + " billion" + (remBillions != 0 ? " " + ConvertInternal(remBillions, andword) : "");
        }

        public static string OrdinalToWords(long num)
        {
            if (num < 20) return OrdinalOnes[num] != "" ? OrdinalOnes[num] : NumberToWords(num) + "th";
            if (num < 100)
            {
                long tens = num / 10;
                long ones = num % 10;
                if (ones == 0) return OrdinalTens[tens];
                return Tens[tens] + " " + OrdinalOnes[ones];
            }
            string cardinal = NumberToWords(num);
            if (cardinal.EndsWith("y")) return cardinal.Substring(0, cardinal.Length - 1) + "ieth";
            if (cardinal.EndsWith("one")) return cardinal.Substring(0, cardinal.Length - 3) + "first";
            if (cardinal.EndsWith("two")) return cardinal.Substring(0, cardinal.Length - 3) + "second";
            if (cardinal.EndsWith("three")) return cardinal.Substring(0, cardinal.Length - 5) + "third";
            if (cardinal.EndsWith("ve")) return cardinal.Substring(0, cardinal.Length - 2) + "fth";
            if (cardinal.EndsWith("e")) return cardinal.Substring(0, cardinal.Length - 1) + "th";
            if (cardinal.EndsWith("t")) return cardinal + "h";
            return cardinal + "th";
        }

        public static string PrepareText(string text)
        {
            text = text.Trim();
            if (string.IsNullOrEmpty(text)) return "";

            // Convert to ASCII
            var sb = new StringBuilder();
            foreach (char c in text)
            {
                if (UnicodeMap.ContainsKey(c)) sb.Append(UnicodeMap[c]);
                else sb.Append(c);
            }
            text = sb.ToString().Normalize(NormalizationForm.FormD);
            sb.Clear();
            foreach (char c in text)
            {
                if (char.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark) sb.Append(c);
            }
            text = sb.ToString();

            text = NormalizeNumbers(text);
            text = NormalizeSpecial(text);
            text = ExpandAbbreviations(text);
            text = ExpandSpecialCharacters(text);
            text = Regex.Replace(text, @"\s+", " ");
            text = text.Replace(" .", ".").Replace(" ?", "?").Replace(" !", "!").Replace(" ,", ",");
            text = Regex.Replace(text, @"\.\.\.+", "[ELLIPSIS]");
            text = Regex.Replace(text, @",+", ",");
            text = Regex.Replace(text, @"[.,]*\.[.,]*", ".");
            text = Regex.Replace(text, @"[.,!]*![.,!]*", "!");
            text = Regex.Replace(text, @"[.,!?]*\?[.,!?]*", "?");
            text = text.Replace("[ELLIPSIS]", "...");

            text = text.Trim();
            if (text.Length > 0 && char.IsLetterOrDigit(text[text.Length - 1])) text += ".";
            if (text.Length > 0 && char.IsLower(text[0])) text = char.ToUpper(text[0]) + text.Substring(1);

            return text;
        }

        private static string NormalizeNumbers(string text)
        {
            text = Regex.Replace(text, @"#(\d)", m => $"number {m.Groups[1].Value}");
            text = Regex.Replace(text, @"(\d)([KMBT])", m => {
                var map = new Dictionary<string, string> { { "k", "thousand" }, { "m", "million" }, { "b", "billion" }, { "t", "trillion" } };
                return $"{m.Groups[1].Value} {map[m.Groups[2].Value.ToLower()]}";
            }, RegexOptions.IgnoreCase);

            for (int i = 0; i < 2; i++)
            {
                text = Regex.Replace(text, @"(\d)([a-z])|([a-z])(\d)", m => {
                    if (m.Groups[1].Success && m.Groups[2].Success) return $"{m.Groups[1].Value} {m.Groups[2].Value}";
                    if (m.Groups[3].Success && m.Groups[4].Success) return $"{m.Groups[3].Value} {m.Groups[4].Value}";
                    return m.Value;
                }, RegexOptions.IgnoreCase);
            }

            text = Regex.Replace(text, @"(\d[\d,]+\d)", m => m.Value.Replace(",", ""));
            text = Regex.Replace(text, @"(^|[^/])(\d\d?[/-]\d\d?[/-]\d\d(?:\d\d)?)($|[^/])", m => m.Groups[1].Value + m.Groups[2].Value.Replace("/", " dash ").Replace("-", " dash ") + m.Groups[3].Value);

            text = Regex.Replace(text, @"\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4}", m => {
                string digits = Regex.Replace(m.Value, @"\D", "");
                if (digits.Length == 10) return $"{string.Join(" ", digits.Substring(0, 3).ToCharArray())}, {string.Join(" ", digits.Substring(3, 3).ToCharArray())}, {string.Join(" ", digits.Substring(6).ToCharArray())}";
                return m.Value;
            });

            text = Regex.Replace(text, @"(\d\d?):(\d\d)(?::(\d\d))?", m => {
                int h = int.Parse(m.Groups[1].Value);
                int min = int.Parse(m.Groups[2].Value);
                int s = m.Groups[3].Success ? int.Parse(m.Groups[3].Value) : 0;
                if (!m.Groups[3].Success)
                {
                    if (min == 0) return h == 0 ? "0" : h > 12 ? $"{h} minutes" : $"{h} o'clock";
                    if (m.Groups[2].Value.StartsWith("0")) return $"{h} oh {m.Groups[2].Value[1]}";
                    return $"{h} {min}";
                }
                string res = "";
                if (h != 0) res = h + " " + (min == 0 ? "oh oh" : m.Groups[2].Value.StartsWith("0") ? $"oh {m.Groups[2].Value[1]}" : min.ToString());
                else if (min != 0) res = min + " " + (s == 0 ? "oh oh" : m.Groups[3].Value.StartsWith("0") ? $"oh {m.Groups[3].Value[1]}" : s.ToString());
                else res = s.ToString();
                return res + " " + (s == 0 ? "" : m.Groups[3].Value.StartsWith("0") ? $"oh {m.Groups[3].Value[1]}" : s.ToString());
            });

            text = Regex.Replace(text, @"£([\d,]*\d+)", m => $"{m.Groups[1].Value.Replace(",", "")} pounds");
            text = Regex.Replace(text, @"\$([\d.,]*\d+)", m => {
                string[] parts = m.Groups[1].Value.Replace(",", "").Split('.');
                int dollars = int.Parse(parts[0]);
                int cents = parts.Length > 1 ? int.Parse(parts[1]) : 0;
                if (dollars > 0 && cents > 0) return $"{dollars} {(dollars == 1 ? "dollar" : "dollars")}, {cents} {(cents == 1 ? "cent" : "cents")}";
                if (dollars > 0) return $"{dollars} {(dollars == 1 ? "dollar" : "dollars")}";
                if (cents > 0) return $"{cents} {(cents == 1 ? "cent" : "cents")}";
                return "zero dollars";
            });

            text = Regex.Replace(text, @"(\d+(?:\.\d+)+)", m => string.Join(" ", m.Value.Replace(".", " point ").ToCharArray().Select(c => c.ToString())));
            text = Regex.Replace(text, @"(\d)\s?\*\s?(\d)", "$1 times $2");
            text = Regex.Replace(text, @"(\d)\s?\/\s?(\d)", "$1 over $2");
            text = Regex.Replace(text, @"(\d)\s?\+\s?(\d)", "$1 plus $2");
            text = Regex.Replace(text, @"(\d)?\s?-\s?(\d)", m => (m.Groups[1].Success ? m.Groups[1].Value : "") + " minus " + m.Groups[2].Value);
            text = Regex.Replace(text, @"(\d+)\/(\d+)", "$1 over $2");
            text = Regex.Replace(text, @"(\d+)(st|nd|rd|th)", m => OrdinalToWords(long.Parse(m.Groups[1].Value)), RegexOptions.IgnoreCase);

            text = Regex.Replace(text, @"\d+", m => {
                long num = long.Parse(m.Value);
                if (num > 1000 && num < 3000)
                {
                    if (num == 2000) return "two thousand";
                    if (num > 2000 && num < 2010) return "two thousand " + NumberToWords(num % 100);
                    if (num % 100 == 0) return NumberToWords(num / 100) + " hundred";
                    return NumberToWords(num, zero: "oh", group: 2);
                }
                return NumberToWords(num);
            });

            return text;
        }

        private static string NormalizeSpecial(string text)
        {
            text = Regex.Replace(text, @"https?:\/\/", "h t t p s colon slash slash ", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"(.) - (.)", "$1, $2");
            text = Regex.Replace(text, @"([A-Z])\.([A-Z])", "$1 dot $2", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"[\(\[\{][^\)\]\}]*[\)\]\}](.)?", m => {
                string result = m.Value.Replace("(", ", ").Replace("[", ", ").Replace("{", ", ").Replace(")", ", ").Replace("]", ", ").Replace("}", ", ");
                if (m.Groups[1].Success && "$.!?, ".Contains(m.Groups[1].Value)) result = result.Substring(0, result.Length - 2) + m.Groups[1].Value;
                return result;
            });
            return text;
        }

        private static string ExpandAbbreviations(string text)
        {
            foreach (var (regex, replacement) in Abbreviations) text = regex.Replace(text, replacement);
            foreach (var (regex, replacement) in CasedAbbreviations) text = regex.Replace(text, replacement);
            return text;
        }

        private static string ExpandSpecialCharacters(string text)
        {
            foreach (var (regex, replacement) in SpecialCharacters) text = regex.Replace(text, replacement);
            return text;
        }

        private static string[] SplitTextIntoSentences(string text)
        {
            // JS: /[^.!?]+[.!?]+|[^.!?]+$/g
            // In C# Regex, we can use Matches.
            var matches = Regex.Matches(text, @"[^.!?]+[.!?]+|[^.!?]+$");
            var sentences = new List<string>();
            foreach (Match m in matches)
            {
                var s = m.Value.Trim();
                if (!string.IsNullOrEmpty(s)) sentences.Add(s);
            }
            return sentences.ToArray();
        }

        private static List<string> SplitTokenIdsIntoChunks(List<long> tokenIds, int maxTokens, SentencePieceWrapper sp)
        {
            var chunks = new List<string>();
            for (int i = 0; i < tokenIds.Count; i += maxTokens)
            {
                int count = Math.Min(maxTokens, tokenIds.Count - i);
                var chunkIds = tokenIds.GetRange(i, count);
                var chunkText = ConvertTokensToString(sp, chunkIds);
                if (!string.IsNullOrEmpty(chunkText)) chunks.Add(chunkText);
            }
            return chunks;
        }

        private static string DecodeIdsToPieceString(SentencePieceWrapper sp, List<long> ids)
        {
            if (ids == null || ids.Count == 0)
                return "";

            // Manually convert each ID back to its piece string using the available IdToPiece method.
            // This array of pieces is then joined to simulate the bulk decode operation.
            var pieces = ids.Select(id => sp.IdToPiece((int)id)).ToArray();

            // SentencePiece pieces are concatenated without extra spaces at this stage.
            return string.Join("", pieces);
        }

        private const string SPIECE_UNDERLINE = "\u2581";

        private static string ConvertTokensToString(SentencePieceWrapper sp, List<long> tokens)
        {
            string outString = "";

            if (tokens.Count > 0)
            {
                // FIX: Use the new helper method for the final segment
                string final = DecodeIdsToPieceString(sp, tokens);
                outString += final;
            }

            // Cleanup
            outString = outString.Replace(SPIECE_UNDERLINE, " ");
            outString = outString.TrimEnd(' ', '\n', '\r', '\t');

            return outString;
        }

        public static List<string> SplitIntoBestSentences(string text, SentencePieceWrapper sp)
        {
            string preparedText = PrepareText(text);
            if (string.IsNullOrEmpty(preparedText)) return new List<string>();

            var sentences = SplitTextIntoSentences(preparedText);
            if (sentences.Length == 0) return new List<string>();

            const int CHUNK_TARGET_TOKENS = 80;
            var chunks = new List<string>();
            string currentChunk = "";

            foreach (var sentenceText in sentences)
            {
                var sentenceTokenIds = new List<long>();
                // Encode
                try
                {
                    var pieces = sp.EncodeToPieces(sentenceText);
                    foreach(var p in pieces) sentenceTokenIds.Add(sp.PieceToId(p));
                }
                catch { continue; }

                int sentenceTokens = sentenceTokenIds.Count;

                if (sentenceTokens > CHUNK_TARGET_TOKENS)
                {
                    if (!string.IsNullOrEmpty(currentChunk))
                    {
                        chunks.Add(currentChunk.Trim());
                        currentChunk = "";
                    }
                    var splitChunks = SplitTokenIdsIntoChunks(sentenceTokenIds, CHUNK_TARGET_TOKENS, sp);
                    chunks.AddRange(splitChunks);
                    continue;
                }

                if (string.IsNullOrEmpty(currentChunk))
                {
                    currentChunk = sentenceText;
                    continue;
                }

                string combined = $"{currentChunk} {sentenceText}";
                // Measure combined tokens
                // Optimization: roughly sum, but better to measure exact if SP behavior with spaces is complex.
                // JS does: tokenizerProcessor.encodeIds(combined).length;
                int combinedTokens = 0;
                try
                {
                     var pieces = sp.EncodeToPieces(combined);
                     combinedTokens = pieces.Count;
                }
                catch { combinedTokens = sentenceTokens + 1000; } // Fallback to force split

                if (combinedTokens > CHUNK_TARGET_TOKENS)
                {
                    chunks.Add(currentChunk.Trim());
                    currentChunk = sentenceText;
                }
                else
                {
                    currentChunk = combined;
                }
            }

            if (!string.IsNullOrEmpty(currentChunk))
            {
                chunks.Add(currentChunk.Trim());
            }

            return chunks;
        }
    }
}
