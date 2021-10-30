import re

subRules = [ r"(?=vc)ccv(?=gg)",     #chan-sti-quer
                r"\bvvc\b",              # aic
                r"\bvvvc\b",             # aies
                r"\bcvvc\b",             # feuj
                r"\bvvcc\b",             # airs
                r"\bvcc\b",              # abc
                r"\bccvc\b",             # ksar
                r"\bcvcc\b",             # jack
                r"\bggvvvvcc\b",         # frayeur
                r"\bcvccgg\b",           # kitsch
                r"\bcvc(?=gu)",          # -lon-guet
                r"gu(?=cv)",             # la-gu-ne
                r"guv(?=cv)",            # lar-gue-ra
                r"guv(?=gg)",            #
                r"guvc\b",               # al-gues
                r"guvc(?=cv)",           #
                r"\\cvvc(?=gu)",          # -four-gu
                r"gu\b",                 #
                r"guv\b",                #
                r"guvv\b",               #
                r"guvvc\b",              #
                r"guvvcc\b",             #
                r"guvc\b",               #
                r"guvcc\b",              #
                r"cvc(?=gu)",            # alanguis
                r"\bvc(?=co)",           # -al-cooli
                r"coo(?=cv)",            # al-coo-lisé
                r"cooc\b",                # alcool
                r"(?=vv)guc\b",           # -ai-gu
                r"(?=vv)guv\b",           # -ai-gu
                r"\bvv(?=gu)",           # -ai-gu
                r"\bvv\b",               # ai
                r"\bvc\b",               # ah
                r"\bv\b",                # a
                r"\bv(?=gg)",            # -i-vraie
                r"\bc\b",                # a
                r"\bco",                 # -co-ordonnée
                r"\bggvvvvc\b",          # -choyait-
                r"ggvc(?=gu)",           # -frin-gue
                r"ggvvv\b",              # le-vreau-
                r"ggvvvv\b",             # lam-proie-
                r"ggvvvvc\b",            # lam-proies-
                r"\bggvvc\b",            # -clair-
                r"\bggvvv\b",            # -cloué-
                r"\bggvcc\b",            # -click-
                r"\bggcv(?=gg)",         # -chro-no
                r"\bggcv(?=cv)",         # -phra-se
                r"ggvvc(?=gg)",          # con-train-dra
                r"cvggc\b",              # -right-
                r"ocv",                  # co-opé-ration
                r"oc",                   # co-or-donnée
                r"\bvc(?=gg)",           # -as- phixiant      
                r"(?=c)ggvvcc\b",        # as-treins-
                r"ggvvccc\b",            # con-traints-
                r"\bvvvcc\b",            # aient
                r"\bggvvvc(?=cv)",       # -crayon-naient
                r"\bggcv(?=cv)",         # -chré-tiens 
                r"\bcvvvvc(?=cv)",       # couaille
                r"\bcvvv(?=gg)",         # -coua-quait
                r"\bcvcc(?=gg)",         # -cons-tuisons
                r"cvgg\b",               # con-cept-
                r"ccv\b",                #chap-ska-
                r"ccvc\b",               #chap-skas-
                r"gg\b",                 # conti-gu-
                r"(?=cv)gg(?=cv)",       # confi-guré- #space
                r"cvvv\b",               # ap-puyé
                r"cvvvv\b",              # ap-puyée
                r"cvvvvc\b",             # ap-puyées
                r"ggvvvcc\b",            # acca-blaient-
                r"cvvccc\b",             # at-teints- 
                r"ggvvc(?=cv)",          # ac-quies-ce
                r"ggvv(?=gg)",           # bi-blio-graphie
                r"ggvvcc\b",             # abs-traits-
                r"ggcv\b",               # algori-thme-
                r"ggcvc\b",              # algori-thmes-
                r"gggvc(?=cv)",          # dé-struc-turer
                r"cvvvv(?=cv)",          # joyeu
                r"cvvvvcc\b",            # accen-tuaient-
                r"cvvcc\b",              # accen-tuant-
                r"cvvv\b",               # accen-tuai-
                r"cvvvcc\b",             # accen-tuai-
                r"cvvvvc\b",             # a-boyait-
                r"cvvvvvcc\b",           # a-boyaient-
                r"cvvvc\b",              # accen-tuais-
                r"cvvvc(?=cv)",          # a-boyan-te
                r"cvvv(?=c)",            # a-boie-ment
                r"\bvcc(?=cv)",          # abs-tient
                r"\bvvc(?=c)",           # -ail-le
                r"cvvc(?=gg)",           # abs-tien-dront
                r"vc(?=gg)",             # -ab-scons
                r"ggvvcc\b",             # -chiant-
                r"ggvvvc\b",             # -truie-
                r"ggvccc\b",             # aca-blants-
                r"ggvvc\b",              # -chien-
                r"ggvv(?=c)",            # -chaî-non
                r"ggv\b",                # va-che-
                r"ggvcc\b",              # quand
                r"ggvc\b",               # chat
                r"ggvc(?=cv)",           # auguille
                r"ggvc(?=cc)",           # ?
                r"ggvc(?=gg)",           # -char-treuse
                r"ggv(?=c)",             # -sta-ble
                r"vc(?=gg)",             # -an-gle
                r"vcc(?=gg)",            # abs-trait
                r"vcc(?=cv)",            # abs-tien
                r"ggvv(?=c)",            # plau-sible
                r"ggvv\b",               # mor-bleu-
                r"ggv(?=gg)",            # re-blo-chon
                r"cvccc\b",              # aba-tants
                r"cvcc\b",               # gar-çons-
                r"cvcc(?=cv)",           # -ping-pong
                r"cvvc(?=c)",            # -ban-que
                r"cvvc\b",               # poissoine-ries-
                r"\bvc(?=cv)",           # -ac-cidentel
                r"\bccv(?=c)",           # -mne-monique
                r"cvc(?=c)",             # -ban-que
                r"cvc\b",                # poi-son-
                r"cvv(?=c)",             # -poi-sonnerie
                r"cvv(?=gg)",            # a-bou-chement
                r"cvc(?=c)",             # poi-son-nerie
                r"cvc(?=gg)",            # rhodo-den-dron
                r"cvv\b",                # poisonne-rie
                r"\bvvv",                # -eau-
                r"\bvv(?=cv)",           # aut
                r"\bv(?=gg)",            # aut
                r"cvv(?=cv)",            # -ton-ton
                r"cv(?=cv)",             # tata
                r"cv\b",                 #
                r"vv(?=gg)",
                r"cv",
                r"v(?=cv)",
                r"gg",
                r"cc",                   # -cm- 
                ]        

# Assemble les règles               
regle = r"("
    
for sR in subRules[:-1]:
    regle += sR + "|"

regle +=  subRules[-1]
regle +=  r")"

to_match_c = [u"bl", u"br", u"ch", u"cl", u"cr", u"dr",
                u"fl", u"fr", u"gh", u"gl", u"gn", u"gr", u"gu", 
                u"kl", u"kr", u"kh", u"kn", u"ph", u"pl", u"pr", u"rh",
                u"qu", u"tr", u"th", u"vr" ] # u"st",u"sc", u"sp", u"pt",,

voyelle  = [u"a", u"à", u"â", u"e", u"é", u"è",
            u"ê", u"i", u"ï", u"î", u"o", u"ô",
            u"u", u"ü", u"û", u"y"]

consonne = [u"b", u"c", u"ç", u"d", u"f", u"g",
            u"h", u"j", u"k", u"l", u"m", u"n",
            u"p", u"q", u"r", u"s", u"t", u"v",
            u"w", u"x", u"z"]

def getConsonneVoyelle(word):
    consonneVoyelleForm = ""
    for i, l in enumerate(word):
        if l in voyelle:
            consonneVoyelleForm += "v"
             
        if l in consonne:
            consonneVoyelleForm += "c"
 
    return consonneVoyelleForm
 
def changeForm(toChange, base):
    """
    cas spéciaux
    """
    # oo - > doit rester oo pour "coordonées"


    for m in to_match_c:
        toChange = replaceOn(toChange, base,  m, "gg" )
     
    to_match_c2 = [ u"oo", u"oé" ]
    for m in to_match_c2:
        toChange = replaceOn(toChange, base,  m, "oo" )
    toChange = replaceOn(toChange, base,  "gu", "gu" )
    return toChange
 
def replaceOn(onThis, base, toFind, by,):
    """
    cherche et remplace des chaînes de caractere dans une chaîne
    en fonction de positions trouvées dans une autre chaine
 
    -Usage-
    replaceOn(getConsonneVoyelle(a), "chat chien",  "ch", "gg" )
    """
 
    pos = [(m.start(), m.end()) for m in re.finditer(toFind, base) ]
 
    for start, end in pos:
        onThis = onThis[0:start] + by + onThis[end:]
     
    return onThis


def getSyllabation(word, count = False):
    consonneVoyelleForm = getConsonneVoyelle(word)
    finalForm = changeForm(consonneVoyelleForm, word)
    #print "final form =", consonneVoyelleForm

    syllabes = [word[gr.start():gr.end()] for gr in re.finditer(regle, finalForm)]
     
    if count:
        return len(syllabes)
    else:
        finalForm
    return syllabes, finalForm