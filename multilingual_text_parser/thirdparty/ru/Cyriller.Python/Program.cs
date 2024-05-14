using System;
using NetMQ;
using NetMQ.Sockets;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cyriller.Model;

namespace Cyriller.Python
{
    class Program
    {
        private const char batch_sep = ';';
        private const char case_sep = ':';
        private const char item_sep = '|';
        private const char all_case_sep = ':';
        private static CyrNounCollection cyrNounCollection;
        private static CyrAdjectiveCollection cyrAdjectiveCollection;
        private static CyrPhrase cyrPhrase;
        private static CyrName cyrName = new CyrName();
        private static CyrNumber cyrNumber = new CyrNumber();

        public class OrdinalNumeral
        {
            private static readonly Dictionary<long, string> Words = new Dictionary<long, string>
            {
                [1] = "первый",
                [2] = "второй",
                [3] = "третий",
                [4] = "четвертый",
                [5] = "пятый",
                [6] = "шестой",
                [7] = "седьмой",
                [8] = "восьмой",
                [9] = "девятый",
                [10] = "десятый",
                [11] = "одиннадцатый",
                [12] = "двенадцатый",
                [13] = "тринадцатый",
                [14] = "четырнадцатый",
                [15] = "пятнадцатый",
                [16] = "шестнадцатый",
                [17] = "семнадцатый",
                [18] = "восемнадцатый",
                [19] = "девятнадцатый",
                [20] = "двадцатый",
                [30] = "тридцатый",
                [40] = "сороковой",
                [50] = "пятидесятый",
                [60] = "шестидесятый",
                [70] = "семидесятый",
                [80] = "восьмидесятый",
                [90] = "девяностый",
                [100] = "сотый",
                [200] = "двухсотый",
                [300] = "трехсотый",
                [400] = "четырехсотый",
                [500] = "пятисотый",
                [600] = "шестисотый",
                [700] = "семисотый",
                [800] = "восемисотый",
                [900] = "девятисотый",
            };
            private static readonly Dictionary<long, string> Exponents = new Dictionary<long, string>
            {
                [1000] = "тысячный",
                [1000000] = "миллионный",
                [1000000000] = "миллиардный",
                [1000000000000] = "триллионный"
            };
            private static readonly Dictionary<long, string> Multipliers = new Dictionary<long, string>()
            {
                [2] = "двух",
                [3] = "трех",
                [4] = "четырех",
                [5] = "пяти",
                [6] = "шести",
                [7] = "седьми",
                [8] = "восьми",
                [9] = "девяти",
                [10] = "десяти",
                [11] = "одиннадцати",
                [12] = "двенадцати",
                [13] = "тринадцати",
                [14] = "четырнадцати",
                [15] = "пятнадцати",
                [16] = "шестнадцати",
                [17] = "семнадцати",
                [18] = "восемнадцати",
                [19] = "девятнадцати",
                [20] = "двадцати",
                [30] = "тридцати",
                [40] = "сорока",
                [50] = "пятьдесяти",
                [60] = "шестьдесяти",
                [70] = "семьдесяти",
                [80] = "восемьдесяти",
                [90] = "девяности",
                [100] = "сто",
                [200] = "двухста",
                [300] = "трехста",
                [400] = "четырехста",
                [500] = "пятиста",
                [600] = "шестиста",
                [700] = "семиста",
                [800] = "восемиста",
                [900] = "девятиста"
            };

            private static CyrNumber Cardinal = new CyrNumber();

            /**
             * @param $number
             * @param string $gender
             * @return array
            */
            public static string Decline(long number, CasesEnum @case, GendersEnum gender = GendersEnum.Masculine, bool plural = false)
            {
                // simple numeral
                if (Words.ContainsKey(number) || Exponents.ContainsKey(number) || number == 0)
                {
                    string word;

                    if (number == 0)
                    {
                        word = "нулевой";
                    }
                    else
                    {
                        if (!Words.TryGetValue(number, out word))
                        {
                            Exponents.TryGetValue(number, out word);
                        }
                    }
                    // special rules for 3
                    if (number == 3)
                    {
                        var prefix = word.Substring(0, word.Length - 2);
                        if (plural)
                        {
                            return prefix + @case switch
                            {
                                CasesEnum.Nominative => "ие",
                                CasesEnum.Genitive => "их",
                                CasesEnum.Dative => "им",
                                CasesEnum.Accusative => "им",
                                CasesEnum.Instrumental => "им",
                                CasesEnum.Prepositional => "их",
                                _ => throw new InvalidOperationException()
                            };
                        }
                        else
                        {
                            return prefix + @case switch
                            {
                                CasesEnum.Nominative => gender == GendersEnum.Masculine ? "ий" : (gender == GendersEnum.Feminine) ? "ья" : "ье",
                                CasesEnum.Genitive => gender == GendersEnum.Feminine ? "ьей" : "ьего",
                                CasesEnum.Dative => gender == GendersEnum.Feminine ? "ьей" : "ьему",
                                CasesEnum.Accusative => gender == GendersEnum.Feminine ? "ью" : "ьего",
                                CasesEnum.Instrumental => gender == GendersEnum.Feminine ? "ьей" : "ьим",
                                CasesEnum.Prepositional => gender == GendersEnum.Feminine ? "ьей" : "ьем",
                                _ => throw new InvalidOperationException()
                            };
                        }
                    }
                    else
                    {
                        string prefix;
                        switch (gender)
                        {
                            case GendersEnum.Masculine:
                                prefix = word.Substring(0, word.Length - 2);
                                return @case switch
                                {
                                    CasesEnum.Nominative => plural ? (prefix + "ые") : word,
                                    CasesEnum.Genitive => prefix + (plural ? "ых" : "ого"),
                                    CasesEnum.Dative => prefix + (plural ? "ым" : "ому"),
                                    CasesEnum.Accusative => plural ? (prefix + "ых") : word,
                                    CasesEnum.Instrumental => prefix + "ым",
                                    CasesEnum.Prepositional => prefix + (plural ? "ом" : "ых"),
                                    _ => throw new InvalidOperationException()
                                };
                            case GendersEnum.Feminine:
                                prefix = word.Substring(0, word.Length - 2);
                                return @case switch
                                {
                                    CasesEnum.Nominative => prefix + (plural ? "ые" : "ая"),
                                    CasesEnum.Genitive => prefix + (plural ? "ых" : "ой"),
                                    CasesEnum.Dative => prefix + (plural ? "ым" : "ой"),
                                    CasesEnum.Accusative => prefix + "ую",
                                    CasesEnum.Instrumental => prefix + (plural ? "ых" : "ой"),
                                    CasesEnum.Prepositional => prefix + (plural ? "ых" : "ой"),
                                    _ => throw new InvalidOperationException()
                                };

                            case GendersEnum.Neuter:
                                prefix = word.Substring(0, word.Length - 2);
                                return @case switch
                                {
                                    CasesEnum.Nominative => prefix + (plural ? "ые" : "ое"),
                                    CasesEnum.Genitive => prefix + (plural ? "ых" : "ого"),
                                    CasesEnum.Dative => prefix + (plural ? "ым" : "ому"),
                                    CasesEnum.Accusative => prefix + (plural ? "ые" : "ое"),
                                    CasesEnum.Instrumental => prefix + "ым",
                                    CasesEnum.Prepositional => prefix + (plural ? "ых" : "ом"),
                                    _ => throw new InvalidOperationException()
                                };
                            default:
                                throw new InvalidOperationException();
                        }
                    }
                }
                // compound numeral
                else
                {
                    string ordinalPart = null;
                    string ordinal_prefix = string.Empty;
                    var result = "";
                    // test for exponents. If smaller summand of number is an exponent, declinate it
                    foreach (var (wordNumber, word) in Exponents.ToArray().OrderByDescending(e => e.Key))
                    {
                        if (number >= wordNumber && (number % wordNumber) == 0)
                        {
                            var count = (long)Math.Floor((decimal)(number / wordNumber)) % 1000;
                            number -= count * wordNumber;
                            foreach (var (multiplier, multiplierWord) in Multipliers.ToArray().OrderByDescending(e => e.Key))
                            //array_reverse(self::$multipliers, true) as $multiplier => $multipliers_word)
                            {
                                if (count >= multiplier)
                                {
                                    ordinal_prefix += multiplierWord;
                                    count -= multiplier;
                                }
                            }
                            ordinalPart = Decline(wordNumber, @case, gender);
                            ordinalPart = ordinal_prefix + ordinalPart;
                            break;
                        }
                    }
                    // otherwise, test if smaller summand is just a number with it's own name
                    if (string.IsNullOrEmpty(ordinalPart))
                    {
                        // get the smallest number with it's own name
                        foreach (var (wordNumber, word) in Words.ToArray())
                        {
                            if (number >= wordNumber)
                            {
                                if (wordNumber <= 9)
                                {
                                    if (number % 10 == 0)
                                    {
                                        continue;
                                    }
                                    // check for case when word_number smaller than should be used (e.g. 1,2,3 when it can be 4 (number: 344))
                                    if ((number % 10) > wordNumber)
                                    {
                                        continue;
                                    }
                                    // check that there is no two-digits number with it's own name (e.g. 13 for 113)
                                    if (Words.ContainsKey(number % 100) && number % 100 > wordNumber)
                                    {
                                        continue;
                                    }
                                }
                                else if (wordNumber <= 90)
                                {
                                    // check for case when word_number smaller than should be used (e.g. 10, 11, 12 when it can be 13)
                                    if ((number % 100) > wordNumber)
                                    {
                                        continue;
                                    }
                                }
                                ordinalPart = Decline(wordNumber, @case, gender);
                                number -= wordNumber;
                                break;
                            }
                        }
                    }
                    // if number has second summand, get cardinal form of it
                    if (number > 0)
                    {
                        var cardinalPart = Cardinal.Decline(number, gender, AnimatesEnum.Inanimated).Get(CasesEnum.Nominative);
                        // make one array with cases and delete 'o/об' prepositional from all parts except the last one
                        result = cardinalPart + " " + ordinalPart;
                    }
                    else
                    {
                        result = ordinalPart;
                    }
                    return result;
                }
            }
        }

        static string process(string input)
        {
            string[] in_data = input.Split(item_sep);
            int len = in_data.Length;

            CasesEnum @case_val = CasesEnum.Nominative;
            GendersEnum @gender_val = GendersEnum.Masculine;
            bool ordinal = false;
            bool plural = false;
            bool all_case = false;

            if (in_data[0].Count(f => f == case_sep) > 0)
            {
                string[] temp = in_data[0].Split(case_sep);
                in_data[0] = temp[0];

                for (int i = 1; i < temp.Length; i++)
                {
                    string num_case = temp[i].Replace("2", "");
                    switch (num_case)
                    {
                        case "им":
                            case_val = CasesEnum.Nominative;
                            break;
                        case "рд":
                            case_val = CasesEnum.Genitive;
                            break;
                        case "дт":
                            case_val = CasesEnum.Dative;
                            break;
                        case "вн":
                            case_val = CasesEnum.Accusative;
                            break;
                        case "тв":
                            case_val = CasesEnum.Instrumental;
                            break;
                        case "пр":
                            case_val = CasesEnum.Prepositional;
                            break;
                        case "all":
                            all_case = true;
                            break;
                        case "мр":
                            gender_val = GendersEnum.Masculine;
                            break;
                        case "жр":
                            gender_val = GendersEnum.Feminine;
                            break;
                        case "ср":
                            gender_val = GendersEnum.Neuter;
                            break;
                        case "мн":
                            plural = true;
                            break;
                        case "ordinal":
                            ordinal = true;
                            break;
                        default:
                            break;
                    }
                }
            }

            if (len == 1)
            {
                string result;
                if (ordinal)
                {
                    in_data[0] = in_data[0].Replace(".", "").Replace(",", "");
                    long value = Convert.ToInt64(in_data[0]);

                    if (all_case)
                    {
                        result = OrdinalNumeral.Decline(value, CasesEnum.Nominative, gender_val, plural);
                        result += all_case_sep + OrdinalNumeral.Decline(value, CasesEnum.Genitive, gender_val, plural);
                        result += all_case_sep + OrdinalNumeral.Decline(value, CasesEnum.Dative, gender_val, plural);
                        result += all_case_sep + OrdinalNumeral.Decline(value, CasesEnum.Accusative, gender_val, plural);
                        result += all_case_sep + OrdinalNumeral.Decline(value, CasesEnum.Instrumental, gender_val, plural);
                        result += all_case_sep + OrdinalNumeral.Decline(value, CasesEnum.Prepositional, gender_val, plural);
                    }
                    else
                    {
                        result = OrdinalNumeral.Decline(value, case_val, gender_val, plural);
                    }
                    return result;
                }
                else
                {
                    decimal value = Convert.ToDecimal(in_data[0]);

                    if (all_case)
                    {
                        CyrResult res = cyrNumber.Decline(value, gender_val, AnimatesEnum.Inanimated);
                        result = String.Join(all_case_sep, res.ToArray());
                    }
                    else
                    {
                        result = cyrNumber.ToString(value, case_val, gender_val, AnimatesEnum.Inanimated);
                    }
                }
                return result;
            }

            if (len == 2)
            {
                string noun = in_data[1];

                if (ordinal)
                {
                    in_data[0] = in_data[0].Replace(".", "").Replace(",", "");
                    long value = Convert.ToInt64(in_data[0]);
                    noun = " " + noun;

                    string result;
                    if (all_case)
                    {
                        result = OrdinalNumeral.Decline(value, CasesEnum.Nominative, gender_val, plural) + noun;
                        result += all_case_sep + OrdinalNumeral.Decline(value, CasesEnum.Genitive, gender_val, plural) + noun;
                        result += all_case_sep + OrdinalNumeral.Decline(value, CasesEnum.Dative, gender_val, plural) + noun;
                        result += all_case_sep + OrdinalNumeral.Decline(value, CasesEnum.Accusative, gender_val, plural) + noun;
                        result += all_case_sep + OrdinalNumeral.Decline(value, CasesEnum.Instrumental, gender_val, plural) + noun;
                        result += all_case_sep + OrdinalNumeral.Decline(value, CasesEnum.Prepositional, gender_val, plural) + noun;
                    }
                    else
                    {
                        result = OrdinalNumeral.Decline(value, case_val, gender_val, plural) + noun;
                    }
                    return result;
                }
                else
                {
                    decimal value = Convert.ToDecimal(in_data[0]);
                    CyrResult result;

                    switch (noun)
                    {
                        case "рубль":
                            result = cyrNumber.Decline(value, new CyrNumber.RurCurrency());
                            break;
                        case "доллар":
                            result = cyrNumber.Decline(value, new CyrNumber.UsdCurrency());
                            break;
                        case "евро":
                            result = cyrNumber.Decline(value, new CyrNumber.EurCurrency());
                            break;
                        case "юань":
                            result = cyrNumber.Decline(value, new CyrNumber.YuanCurrency());
                            break;
                        default:
                            CyrNoun item = cyrNounCollection.Get(noun, out CasesEnum _, out NumbersEnum _);
                            result = cyrNumber.Decline(value, new CyrNumber.Item(item));
                            break;
                    }

                    if (all_case)
                    {
                        return String.Join(all_case_sep, result.ToArray());
                    }
                    else
                    {
                        switch (case_val)
                        {
                            case CasesEnum.Nominative:
                                return result.Именительный;
                            case CasesEnum.Genitive:
                                return result.Родительный;
                            case CasesEnum.Dative:
                                return result.Дательный;
                            case CasesEnum.Accusative:
                                return result.Винительный;
                            case CasesEnum.Instrumental:
                                return result.Творительный;
                            case CasesEnum.Prepositional:
                                return result.Предложный;
                            default:
                                return result.Именительный;
                        }
                    }

                }
            }

            return "error";
        }

        static void Main(string[] args)
        {
            long port = 5556;

            if (args.Length == 1)
            {
                port = Convert.ToInt64(args[0]);
            }

            Console.WriteLine("Start Cyriller on port " + port.ToString());

            cyrNounCollection = new CyrNounCollection();

            using (var server = new ResponseSocket())
            {
                server.Bind("tcp://*:" + port.ToString());

                while (true)
                {
                    string msg = server.ReceiveFrameString();
                    if (msg == "exit") break;

                    string[] batch = msg.Split(batch_sep);
                    for (int i = 0; i < batch.Length; i++)
                    {
                        string result = "error";
                        try
                        {
                            result = process(batch[i]);
                        }
                        catch { }
#if DEBUG
                        Console.WriteLine(batch[i] + " -> " + result);
#endif
                        batch[i] = result;
                    }

                    server.SendFrame(string.Join(batch_sep, batch));
                }

                server.Close();
            }
        }

    }

}
