import pytest

from multilingual_text_parser.data_types import Doc
from multilingual_text_parser.parser import TextParser

text_processor = TextParser(lang="EN")

testdata = [
    # numbers
    (
        "10,000 256 4358",
        "ten thousand two hundred and fifty six four thousand three hundred and fifty eight",
    ),
    ("−3,000", "minus three thousand"),
    ("+5", "plus five"),
    (
        "4.5 -3.1 1,000.12",
        "four point five minus three point one one thousand point one two",
    ),
    ("21st 42nd 6th 1,000,000th", "twenty first forty second sixth one millionth"),
    # ("60s 100s bc 52's", "sixties one hundreds b c fifty two's"),
    ("3/4 2 1/2 −7 2/3", "three quarters two one half minus seven two thirds"),
    (
        "0123 924-51-0387",
        "o one two three nine two four five one o three eight seven",
    ),
    ("Elizabeth II", "elizabeth second"),
    ("I am the 1st, you are the 2nd", "i am the first, you are the second"),
    # measures
    # ("1h2m30s", "one hour two minutes thirty seconds"),
    ("25 MPH", "twenty five miles per hour"),
    # ("2.6 GHz", "two point six gigahertz"),
    ("-0.01%", "minus zero point zero one percent"),
    ("It's −3°C today", "it's minus three degrees celsius today"),
    # ("50¢ 40 km/h", "fifty cents forty kilometers per hour"),
    ("90°", "ninety degrees"),
    # currency
    ("$10", "ten dollars"),
    ("£5", "five pounds"),
    ("€20", "twenty euros"),
    ("£5.27", "five pounds twenty seven pence"),
    # ("€ 20 000", "twenty thousand euros"),
    # ("USD5.27", "five u s dollars and twenty seven cents"),
    # ("GBP 1,000", "one thousand pounds"),
    ("¥5.27", "five yen twenty seven sen"),
    # ("¥1 million", "one million yen"),
    # ("CHF6M", "six million swiss francs"),
    # ("C$ 2.3 mn", "two point three million canadian dollars")
    # time
    ("1:59 2:00", "one fifty nine two o'clock"),
    ("1:59am 2 AM", "one fifty nine a m two a m"),
    ("10:25:30", "ten hours twenty five minutes and thirty seconds"),
    # ("5\'30\"", "five minutes and thirty seconds"),
    ("07:53:10 A.M.", "seven hours fifty three minutes and ten seconds a m"),
    # ("5m30s", "five minutes and thirty seconds"),
    # ("3h10m", "three hours and ten minutes"),
    # geographic coordinates
    # ("74.3°W", "seventy four point three degrees west"),
    # ("13°45\' N", "thirteen degrees and forty five minutes north"),
    # date
    ("12/31/1999", "december thirty first nineteen ninety nine"),
    ("10-25-99", "october twenty fifth ninety nine"),
    # ("Dec/31/1999", "december thirty first nineteen ninety nine"),
    # ("April-25-1999", "april twenty fifth nineteen ninety nine"),
    # ("12/may/1995", "the twelfth of may nineteen ninety five"),
    # ("12-Apr-2007", "april twelfth two thousand seven"),
    # ("20.3.2011", "march twentieth twenty eleven"),
    ("2007/01/01", "january first two thousand seven"),
    # ("2007-Jan-01", "january first two thousand seven"),
    # ("2007-January-01", "january first two thousand seven"),
    # ("June 2", "june second"),
    # ("Aug. 5, 1921", "august fifth nineteen twenty one"),
    # ("arrive on 3/4", "arrive on march fourth"),
    # ("1063 A.D.", "ten sixty three a d"),
    # range
    # ("ages 3–5", "ages three to five"),
    # ("40Hz–20kHz", "forty hertz to twenty kilohertz"),
    # ("June 15-20", "june fifteenth to twentieth"),
    ("1939-1945", "nineteen thirty nine to nineteen forty five"),
    # abbreviations
    ("i.e. Mr T vs bros Inc.", "that is. mister t versus brothers incorporated"),
    # initialisms
    # ("N.Y.P.D.", "n y p d"),
    ("BBC", "b b c"),  # ?
    ("pwq", "pwq"),  # ?
    # street address
    (
        "159 W. Popplar Ave., Ste. 5, St. George, CA 12345",
        "one hundred and fifty nine west popplar avenue. suite. five, street. george, c a twelve thousand three hundred and forty five",
    ),
    # telephone number
    # ("(978) 555-2345", "nine seven eight five five five two three four five"),
    # ("1-800-555-1234 ex. 10", "one eight hundred five five five one two three four extension one zero"),
    # apostrophe
    ("it's time", "it's time"),
    ("Jenny's speech", "jenny's speech"),
    ("The girls’ bedroom", "the girls bedroom"),
]


@pytest.mark.parametrize("text, expected", testdata)
def test_normalizer(text, expected):
    doc = Doc(text)
    doc = text_processor.process(doc)
    assert doc.text.rstrip(".") == expected
