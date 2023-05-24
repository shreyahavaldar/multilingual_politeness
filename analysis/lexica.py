import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import argparse
import time, sys
import re

hedges = [
    "think", "thought", "thinking", "almost",
    "apparent", "apparently", "appear", "appeared", "appears", "approximately", "around",
    "assume", "assumed", "certain amount", "certain extent", "certain level", "claim",
    "claimed", "doubt", "doubtful", "essentially", "estimate",
    "estimated", "feel", "felt", "frequently", "from our perspective", "generally", "guess",
    "in general", "in most cases", "in most instances", "in our view", "indicate", "indicated",
    "largely", "likely", "mainly", "may", "maybe", "might", "mostly", "often", "on the whole",
    "ought", "perhaps", "plausible", "plausibly", "possible", "possibly", "postulate",
    "postulated", "presumable", "probable", "probably", "relatively", "roughly", "seems",
    "should", "sometimes", "somewhat", "suggest", "suggested", "suppose", "suspect", "tend to",
    "tends to", "typical", "typically", "uncertain", "uncertainly", "unclear", "unclearly",
    "unlikely", "usually", "broadly", "tended to", "presumably", "suggests",
    "from this perspective", "from my perspective", "in my view", "in this view", "in our opinion",
    "in my opinion", "to my knowledge", "fairly", "quite", "rather", "argue", "argues", "argued",
    "claims", "feels", "indicates", "supposed", "supposes", "suspects", "postulates"
]

positive_words = []
negative_words = []
taboo_words = []
best_wishes_words = []
praise_words = []
emergency_words = []


# politeness_strategies = {
#     "please": [r"(?<=\w\s)\b(please|pls|plz)\b"],
#     "please_start": [r"^\bplease\b", r"^\W*please|pls|plz\b"],
#     "hedge": [rf"\b{hedge}\b" for hedge in hedges],
#     "indirect_btw": [r"\b(btw|by the way)\b"],
#     "factuality": [r"\bin fact\b", r"\bthe (point|reality|truth)\b", r"\b(really|actually|honestly|surely)\b", r"\b(((in|as a matter of) (fact|truth))|actually)\b"],
#     "deference": [r"\b(great|good|nice|interesting|cool|excellent|awesome)\b"],
#     "gratitude": [r"\b(thank|thanks)\b", r"\bi appreciate\b", r"\b(thanks|thx|thank you|i appreciate)\b"],
#     "apologetic": [r"\b(sorry|woops|oops)\b", r"\bi apologize\b", r"\b(forgive|excuse) me\b", r"\b(my (bad|apologies|apology|fault)|sorry|sry)\b"],
#     "first_person_plural": [r"\b(we|our|us|ourselves)\b", r"\b(we|our|ours|us)\b"],
#     "first_person_singular": [r"\b(i|my|mine|myself)\b", r"\b(i|my|mine|me)\b"],
#     "first_person_start": [r"^(i|my|mine|myself)\b", r"^\W*i\b"],
#     "second_person": [r"\b(you|your|yours|yourself)\b", r"\b(u|you)\b"],
#     "second_person_start": [r"^(you|your|yours|yourself)\b", r"^\W*(u|you)\b"],
#     "greeting": [r"^(hi|hello|hey)\b", r"\b(hello|he+y|hi+|aloha)\b"],
#     "direct_question": [r"\b(what|why|who|how)\b"],
#     "direct_start": [r"^(so|then|and|but|or)\b", r"^\W*so+\b"],
#     "has_positive": [rf"\b{word}\b" for word in positive_words],
#     "has_negative": [rf"\b{word}\b" for word in negative_words],
#     "subjunctive": [r"\b(could|would) you\b", r"\b[cw]ould u|you\b"],
#     "indicative": [r"\b(can|will) you\b", r"\b(can|will) (u|you)\b"],
#     "you_honorific": [r"\b(ur|your) (honour|honor|majesty|highness)\b"],
#     "you_direct": [r"\b(u|you)\b"],
#     "taboo": [rf"\b{word}\b" for word in taboo_words],
#     "best_wishes": [rf"\b{word}\b" for word in best_wishes_words],
#     "praise": [rf"\b{word}\b" for word in praise_words],
#     "emergency": [rf"\b{word}\b" for word in emergency_words],
#     "promise": [r"\b(must|definitely|sure|definite|surely|certainly|promise)\b"],
#     "ingroup_ident": [r"\b(mate|bro|homie)\b"],
#     "together": [r"\btogether\b"],
#     "start_question": [r"^\W*what|where|why|when|how\b"]
# }

politeness_strategies = {
    "gratitude": [r"\b(thanks|thx|thank you|i appreciate)\b"],
    "deference": [r"\b(great|good|nice|interesting|cool|excellent|awesome)\b"],
    "greeting": [r"^(hi|hello|hey)\b", r"\b(hello|he+y|hi+)\b"],
    "has_positive": [rf"\b{word}\b" for word in positive_words],
    "has_negative": [rf"\b{word}\b" for word in negative_words],
    "apologetic": [r"\b(sorry|woops|oops)\b", r"\bi apologize\b", r"\b(forgive|excuse) me\b", r"\b(my (bad|apologies|apology|fault)|sorry|sry)\b"],
    "please": [r"(?<=\w\s)\b(please|pls|plz)\b"],
    "please_start": [r"^\W*please|pls|plz\b"],
    "indirect_btw": [r"\b(btw|by the way)\b"],
    "direct_question": [r"\b(what|where|why|who|when|how)\b"],
    "direct_start": [r"^(so|then|and|but|or)\b", r"^\W*so+\b"],
    "subjunctive": [r"\b(could|would) (u|you)\b"],
    "indicative": [r"\b(can|will) (u|you)\b"],
    "first_person_start": [r"^(i|my|mine|myself)\b", r"^\W*i\b"],
    "first_person_plural": [r"\b(we|our|ours|us|ourselves)\b"],
    "first_person_singular": [r"\b(i|my|mine|myself|me)\b"],
    "second_person": [r"\b(you|your|yours|yourself)\b", r"\b(u|you)\b"],
    "second_person_start": [r"^(you|your|yours|yourself)\b", r"^\W*(u|you)\b"],
    "hedge": [rf"\b{hedge}\b" for hedge in hedges],
    "factuality": [r"\bin fact\b", r"\bthe (point|reality|truth)\b", r"\b(really|actually|honestly|surely)\b", r"\b(((in|as a matter of) (fact|truth))|actually)\b"],

    "emergency":[r"\b(right now|rn|as soon as possible|asap|immediately|hurry up|straightaway|at once)\b"],
    "ingroup_ident": [r"\b(mate|bro|homie)\b"],
    "praise": [r"\b(awesome|outstanding|excellent|great|good|neat|remarkable|fantastic|super|beautiful|bravo|incredible)\b"],
    "promise": [r"\b(must|definitely|sure|definite|surely|certainly|promise)\b"],
    "together": [r"\btogether\b"],
}


positive_words_spanish = []
negative_words_spanish = []
taboo_words_spanish = []
best_wishes_words_spanish = []
praise_words_spanish = []
emergency_words_spanish = []


hedges_spanish = [
    "a menudo", "a mi parecer", "a veces", "adivino", "afirma", "afirmar", "afirmó", "algo", "algunas veces", "alrededor",
    "ambiguamente", "ampliamente", "aparece", "aparecer", "apareció","aparente", "aparentemente", "aproximadamente", "argumenta",
    "argumentar", "argumentó", "asegurar", "aseguró", "asumo", "asumí","bastante", "calculado", "calcular", "casi", "cerca",
    "cierta cantidad", "cierto grado", "cierto nivel", "cierto número","completamente", "con frecuencia", "creo", "debería", "desconfía",
    "desde esta opinión", "desde esta perspectiva","desde este punto de vista", "desde mi perspectiva",
    "desde mi punto de vista", "desde nuestra perspectiva", "discute","discutir", "discutió", "duda", "dudoso", "en conjunto",
    "en esta opinión", "en general", "en gran medida","en la mayoría de las instancias","en la mayoría de las situaciones", "en la mayoría de los casos",
    "en mi opinión", "en nuestra creencia", "en nuestra opinión","en su mayoría", "esencialmente", "estimado", "estimar",
    "evidente", "evidentemente", "frecuentemente", "generalmente","hasta cierto punto", "hasta donde sé", "improbable",
    "inciertamente", "incierto", "indica", "indicar", "indicó","la mayoría de las veces", "mayormente", "moderadamente",
    "más bien", "más o menos", "no claro", "parece", "pareció","pensaba", "pensando", "pensé", "pienso", "plausible",
    "plausiblemente", "poco claramente", "poco claro", "poco probable","podría", "posible", "posiblemente", "postula", "postulado",
    "postular", "presumiblemente", "presumo", "presumí","principalmente", "probable", "probablemente", "puede",
    "puede ser", "quizás", "relativamente", "según mi conocimiento","sentí", "señala", "siente", "siento", "sospecha", "sospechar",
    "sospecho", "sostiene", "sugerido", "sugerir", "sugiere","sugirió", "supone", "suponer", "supongo", "supuesto", "tal vez",
    "tendía a", "tiende a", "tienden a", "típicamente", "típico","un poco", "usual", "usualmente"
]

# politeness_strategies_spanish = {
#     "please": [r"(?<=\w\s)\b(por favor)\b"],
#     "please_start": [r"^\bpor favor\b"],
#     "hedge": [rf"\b{hedge}\b" for hedge in hedges_spanish],
#     "indirect_btw": [r"\ba propósito\b", r"\bpor cierto\b"],
#     "factuality": [r"\ben realidad\b", r"\bde hecho\b", r"\b(la verdad|la realidad)\b", r"\b(en realidad|de hecho|la verdad es que|seguro|realmente|verdaderamente|honestamente|seguramente)\b"],
#     "deference": [r"\b(excelente|bueno|interesante|genial|fresco|asombroso)\b"],
#     "gratitude": [r"\b(gracias|agradecido|agradezco)\b", r"\b(aprecio|agradezco)\b", r"\b(gracias|agradecido|te agradezco)\b"],
#     "apologetic": [r"\b(lo siento|perdón|disculpa)\b", r"\b(pido disculpas|me disculpo)\b", r"\b(perdona|excusa|disculpe|perdone) me\b", r"\b(mis disculpas|mi culpa|fallo mio|lo siento)\b"],
#     "first_person_plural": [r"\b(nosotros|nuestro|nos|nosotras|nosotros mismos)\b"],
#     "first_person_singular": [r"\b(yo|mío|mía|mi|yo mismo)\b"],
#     "first_person_start": [r"^(yo|mío|mía|mi|yo mismo)\b"],
#     "second_person": [r"\b(tú|tu|tuyo|tuya|usted|su|suyo|suya|tú mismo|u|you)\b"],
#     "second_person_start": [r"^(tú|tu|tuyo|tuya|usted|su|suyo|suya|tú mismo|u|you)\b"],
#     "greeting": [r"^(hola|buenas|oye)\b"],
#     "direct_question": [r"\b(qué|quién|cómo|por qué|dónde|what|why|who|how)\b"],
#     "direct_start": [r"^(entonces|y|pero|o|so|then|and|but|or)\b"],
#     # "has_positive": [rf"\b{word}\b" for word in positive_words_spanish],
#     # "has_negative": [rf"\b{word}\b" for word in negative_words_spanish],
#     "subjunctive": [r"\b(podrías|podría|could)\b", r"\b[cw]ould u|you\b"],
#     "indicative": [r"\b(puedes|puede|can|will)\b", r"\b(can|will) u|you\b"],
#     "you_honorific": [r"\b(su|vuestra|your|ur) (señoría|majestad|alteza|honor|honour|majesty|highness)\b"],
#     "you_direct": [r"\b(tú|usted|you|u)\b"],
#     # "taboo": [rf"\b{word}\b" for word in taboo_words_spanish],
#     # "best_wishes": [rf"\b{word}\b" for word in best_wishes_words_spanish],
#     # "praise": [rf"\b{word}\b" for word in praise_words_spanish],
#     # "emergency": [rf"\b{word}\b" for word in emergency_words_spanish],
#     "promise": [r"\b(debo|definitivamente|seguro|ciertamente|prometo)\b"],
#     "ingroup_ident": [r"\b(amigo|compa|colega|bro|homie)\b"],
#     "together": [r"\bjuntos\b"],
#     "start_question": [r"^\W*(qué|dónde|porqué|cuándo|cómo|what|where|why|when|how)\b", r"^\bpor qué\b"],
# }


politeness_strategies_spanish = {
    "gratitude": [r"\b(gracias|aprecio|agradezco|thanks|thank you)\b"],
    "deference": [r"\b(gran|bueno|bonito|interesante|genial|excelente|increíble|fresco|asombroso)\b"],
    "greeting": [r"^(hola|hey)\b", r"\b(hola|o+la)\b"],
    "has_positive": [rf"\b{word}\b" for word in positive_words_spanish],
    "has_negative": [rf"\b{word}\b" for word in negative_words_spanish],
    "apologetic": [r"\b(lo siento|perdón|disculp*)\b", r"\b(perdona|excusa|disculpe|perdone) me\b", r"\b(mis disculpas|mi culpa|fallo mio)\b"],
    "please": [r"(?<=\w\s)\b(por favor|pls|plz|please)\b"],
    "please_start": [r"^\W*por favor|pls|plz|please\b"],
    "indirect_btw": [r"\b(a propósito|por cierto)\b"],
    "direct_question": [r"\b(qué|dónde|por qué|porqué|quién|cuándo|cómo)\b"],
    "direct_start": [r"^(así que|entonces|y|pero|o)\b", r"^\W*así que+\b"],
    "subjunctive": [r"\b(podría*|haría*)\b", r"\b(could|would) u|you\b"],
    "indicative": [r"\b(puede*|hará*)\b", r"\b(can|will) u|you\b"],
    "first_person_start": [r"^(yo|mi|mío|mía|yo mismo|me)\b", r"^\W*yo\b"],
    "first_person_plural": [r"\b(nosotros|nuestro|nuestros|nos|nosotros mismos)\b"],
    "first_person_singular": [r"\b(yo|mi|mío|mía|yo mismo|me)\b"],
    "second_person": [r"\b(tú|tu|tuyo|tuya|usted|su|suyo|suya|tú mismo|u|you)\b"],
    "second_person_start": [r"^(tú|tu|tuyo|tuya|usted|su|suyo|suya|tú mismo|u|you)\b"],
    "hedge": [rf"\b{hedge}\b" for hedge in hedges_spanish],
    "factuality": [r"\b(la verdad|la realidad)\b", r"\b(en realidad|de hecho|la verdad es que|seguro|realmente|verdaderamente|honestamente|seguramente)\b"],

    "emergency": [r"\b(inmediatamente|ahora mismo|lo antes posible|lo más pronto posible|urgente|rápido|inmediato|al instante)\b"],
    "ingroup_ident": [r"\b(colega|hermano|compañero)\b"],
    "praise": [r"\b(asombroso|destacado|excelente|genial|bueno|limpio|notable|fantástico|súper|hermoso|bravo|increíble)\b"],
    "promise": [r"\b(debo|definitivamente|seguro|definitivo|seguramente|ciertamente|prometo)\b"],
    "together": [r"\bjuntos\b"],

    "you_direct": [r"\b(tú|you|u)\b"],
    "you_honorific": [r"\b((usted|señoría|majestad|alteza|honor|honour|majesty|highness)\b"],
}


taboo_chinese = ["拷", "靠", "操", "艹", "草", "cao", "我擦", "擦嘞", "干", "呸", "夭寿", 
                 "他妈", "他妹的", "你妈", "你妹", "nm", "tm", "去你的", "他奶奶的", "tnnd", 
                 "妈蛋", "妈的", "md", "该死", "靠背", "靠杯", "白目", "白痴", "人渣", "王八蛋", 
                 "怪胎", "孬种", "畜生", "淫妇", "混蛋", "混蛋", "魂淡", "龟孙", "笨蛋", "智障", 
                 "傻瓜", "蠢猪", "蠢狗", "傻狗", "窝囊废", "废物", "泼妇", "骚货", "骚逼", "贱人", 
                 "贱货", "荡妇", "杂种", "坏蛋", "烂货", "傻帽", "250", "贰佰伍", "二货", "2B", 
                 "二百五", "SB", "傻逼", "傻B", "煞笔", "沙比", "混账", "婊子", "脑残", "米田共", 
                 "屁", "屎", "屌", "粪", "尿", "死"]

hedges_chinese = ["可", "可以", "能", "不能", "应", "应该", "需", "会", "不会", "将", 
                  "一些", "几乎", "上下", "左右", "尽可能", "多", "少数", "多数", 
                  "验证", "按时", "表明", "推测", "判断", "猜测", "猜", "估计", "大概", 
                  "没准", "也许", "或许", "或者", "可能", "似乎", "说不定", "少许", "稍微", 
                  "一点儿", "一点", "一丁点", "一丁点儿", "稍稍", "少量"]

positive_words_chinese = []
negative_words_chinese = []

politeness_strategies_chinese = {
    "gratitude": re.compile(r"[感谢重鸣]谢"),
    "deference": re.compile(r"([真好]?(厉害|棒|强|牛|美|漂亮))|(干[得的]真?(好|漂亮))"),
    "greeting": re.compile(r"(嗨|哈喽|哈罗|嘿|你好|早上好|上午好|中午好|下午好|晚上好|晚安|安安|安好|你们好|大家好|久违了|好久不见)"),    "has_positive": re.compile(r"({})".format("|".join(positive_words_chinese))),
    "has_negative": re.compile(r"({})".format("|".join(negative_words_chinese))),
    "apologetic": re.compile(r"(对不起|对不住|抱歉|骚瑞|不好意思|很遗憾)"),
    "please": re.compile(r"(请|拜托|帮忙)"),
    "indirect_btw": re.compile(r"(对了|话说|说起来|顺便说一下|顺带提一句|顺带一提|顺带问一下|附带说一下)"),
    "direct_question": re.compile(r"^\W*([为凭]什么|几|哪|多少|怎|谁|咋|什么时候|何时|为何|如何|为什么|为啥|啥|干嘛|干啥)"),
    "subjunctive": re.compile(r"[你您](?P<A>[可想觉要愿意希望]|想要|渴望|期望|意愿|愿意])(不|没)(?P=A)"),
    "indicative": re.compile(r"[你您][是可想觉要愿意希望能能够得].+?[吗呢呀？]"),
    "first_person_plural": re.compile(r"(我们|咱们|咱们俩|咱俩|我们大家)"),
    "first_person_singular": re.compile(r"(我|俺)"),
    "second_person": re.compile(r"你"),
    "hedge":  re.compile(r"({})".format("|".join(hedges_chinese))),
    "factuality": re.compile(r"(其实|说实话|讲真)"),

    "emergency": re.compile(r"(赶紧|立刻|马上|赶快|紧急|尽快|立马|迅速|急需|紧迫|亟需)"),
    "ingroup_ident": re.compile(r"(咱|咱们|俺们|我们)"),
    "praise": re.compile(r"([真好]?(厉害|棒|强|牛|美|漂亮))|(干[得的]真?(好|漂亮))"),
    "promise": re.compile(r"(一定|肯定|绝对|没错|必|保证|确凿)"),
    "together": re.compile(r"一([起齐同]|块儿)"),
    
    "you_direct": re.compile(r"你"),
    "you_honorific": re.compile(r"(您|阁下|先生|女士)"),
    "taboo": re.compile(r"({})".format("|".join(taboo_chinese))),
}

positive_words_japanese = []
negative_words_japanese = []

taboo_japanese = ["くそ", "ちくしょう", "ばか", "あほ", "まぬけ", "ぼけ", "てめえ", "きさま", "このやろう", 
                  "くそったれ", "だめだこいつ", "はげ", "ぶす", "ぶさいく", "でぶ", "しんじまえ", 
                  "きえろ", "ざけんな", "くたばれ", "くそくらえ", "しね", "しにおちろ", "はずかしい", 
                  "ふざけるな", "ふざけんな", "どけ", "うるさい", "だまれ", "うざい", "いじめる", 
                  "さわぐ", "ひどい", "いたずら", "ぬけめ", "へたくそ", "ぐうたら"]

hedges_japanese = ["かもしれない", "だろう", "おそらく", "もしかすると", "ひょっとすると", "たぶん", "いくらか", 
                   "ほとんど", "ほぼ", "少々", "ちょっと", "かなり", "大体", "大概", "一部", "多分", 
                   "まあ", "ある程度", "少しだけ", "少し", "多くの", "少ない", "大多数", "少数", "多数", 
                   "なんとなく", "なんとか", "それなりに", "ある意味で", "ある程度まで", "ある種"]

politeness_strategies_japanese = {
    "gratitude": re.compile(r"(感謝|お礼|いた|謝意|おかげ|ありがたい|有難う|有り難う|感謝しています|お礼申し上げます|お礼を言います|感謝の意を表します)"),
    "deference": re.compile(r"([すごい]?[凄厲棒強牛美綺]|すごく[上手い綺麗]|立派)"),
    "greeting": re.compile(r"(こんにちは|ハロー|やあ|もしもし|おはよう|こんばんは|ねえ)"),
    "has_positive": re.compile(r"({})".format("|".join(positive_words_japanese))),
    "has_negative": re.compile(r"({})".format("|".join(negative_words_japanese))),
    "apologetic": re.compile(r"(すみません|ごめんなさい|申し訳ありません|申し訳ない|恐れ入ります|残念です)"),
    "please": re.compile(r"(お願い|頼む|手伝って|どうか|お願いします)"),
    "indirect_btw": re.compile(r"(ところで|ちなみに)"),
    "direct_question": re.compile(r"^\W*(なぜ|いくつ|どれくらい|どのくらい|どのぐらい|どうやって|誰|どう|何故|いくら|どの程度|どのぐらいの量|どのように|どの方法で|誰が|どのような)"),
    "subjunctive": re.compile(r"[あなたが](?P<A>[思考希望欲願望]|望む|希望する|願う|欲しい)(しない|ない)(?P=A)"),
    "indicative": re.compile(r"[あなたきみあんた][はも] .+?[ですか？]"),
    "first_person_plural": re.compile(r"(私たち|僕たち|俺たち|我々|われわれ)"),
    "first_person_singular": re.compile(r"(私|わたくし|わし|あたし|俺|おれ|僕|ぼく)"),
    "second_person": re.compile(r"あなた"),
    "hedge": re.compile(r"({})".format("|".join(hedges_japanese))),
    "factuality": re.compile(r"(実際|正直に言って|本当に)"),

    "emergency": re.compile(r"(急いで|すぐに|即座に|早く|緊急|できるだけ早く|即刻|大至急|至急|早急に|迅速に|速やかに|切実に)"),
    "ingroup_ident": re.compile(r"(私たち|我々|僕たち|俺たち|われわれ|うちら|我ら)"),
    "praise": re.compile(r'(すごい|上手|美しい|きれい|素晴らしい|優れた|立派な|かっこいい|最高|素晴らしい|感動的|驚くべき|素敵な|嬉しい|すばらしい|ですね)'),
    "promise": re.compile(r"(必ず|確かに|絶対に|間違いなく|確実に)"),
    "together": re.compile(r"(一緒に|一同)"),

    "you_direct": re.compile(r"(あなた|きみ|あんた)"),
    "you_honorific": re.compile(r"(あなた様|あなたさま|貴方様|貴方さま)"),
    "taboo": re.compile(r"({})".format("|".join(taboo_japanese))),
}




def generate_lexica():
    lexica_dict = {}
    lexica_dict["English"] = politeness_strategies
    lexica_dict["Chinese"] = politeness_strategies_chinese
    lexica_dict["Japanese"] = politeness_strategies_japanese
    lexica_dict["Spanish"] = politeness_strategies_spanish
    return lexica_dict

