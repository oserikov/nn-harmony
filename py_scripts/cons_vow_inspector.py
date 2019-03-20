import sys

vowels = [c for c in "aeiıoöüu"]
alphabet = [c for c in "abcçdefgğhıijklmnoöprsştuüvyz"]
consonants = list(set(alphabet).difference(set(vowels)))


consonants_followed_by_consonants = {key: 0 for key in consonants}
consonants_cnt  = {key: 0 for key in consonants}

for line in sys.stdin:
    word = line.strip()
    for idx,_ in enumerate(word[1:]):
        if word[idx] in consonants and word[idx-1] in consonants:
            consonants_followed_by_consonants[word[idx-1]] += 1

    for char in word:
        if char in consonants:
            consonants_cnt[char] += 1

for consonant in consonants:
    consonants_followed_by_consonants[consonant] = consonants_followed_by_consonants[consonant]/consonants_cnt[consonant]

for entry in sorted(consonants_followed_by_consonants.items(), key=lambda kv: kv[1], reverse=True):
    print(entry[0], entry[1], consonants_cnt[entry[0]])
