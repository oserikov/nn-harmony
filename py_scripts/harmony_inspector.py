import sys


features = {
    "front": {
        "+": [char for char in "eiöü"],
        "-": [char for char in "aıou"]
    },
    "round": {
        "+": [char for char in "oöuü"],
        "-": [char for char in "aeıi"]
    }
}

vowels = [char for char in "aıoueiöü"]

features_by_char = {key: [] for key in vowels}
for char in vowels:
    for feature in features.keys():
        for feature_value in features[feature].keys():
            for featured_vowel in features[feature][feature_value]:
                features_by_char[featured_vowel].append(feature_value + feature)

for char in vowels:
    features_by_char[char] = set(features_by_char[char])

all_the_features = set()
for fs in features_by_char.values():
    for f in fs:
        all_the_features.add(f)

features_qtty = {f: 0 for f in all_the_features}


def check_for_harmony(word):
    vowels_in_word = [char for char in word if char in vowels]

    harm_features = all_the_features
    for vowel in vowels_in_word:
        harm_features = harm_features.intersection(set(features_by_char[vowel]))

    if harm_features:
        for feature in harm_features:
            features_qtty[feature] += 1
        print(word, " ".join(harm_features))


def main():
    for line in sys.stdin:
        word = line.strip()
        check_for_harmony(word)


main()

print(features_qtty)
