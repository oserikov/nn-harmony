export PYTHONIOENCODING="UTF-8"
sh_scripts/lexemes_generator.sh |\
sh_scripts/phonemes_sequences_extractor.sh |\
sh_scripts/turkish_onehot_encoder.sh |\
python3 dynet_training.py 4
