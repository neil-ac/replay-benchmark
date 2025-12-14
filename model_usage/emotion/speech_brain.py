"""
pip install git+https://github.com/speechbrain/speechbrain.git@develop

https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP


'neu' => 0
'ang' => 1
'hap' => 2
'sad' => 3

other data to try? https://huggingface.co/datasets/AbstractTTS/IEMOCAP

"""


from speechbrain.inference.interfaces import foreign_class

classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

out_prob, score, index, text_lab = classifier.classify_file("audio/sad.wav")  # neu
# out_prob, score, index, text_lab = classifier.classify_file("audio/surprised.wav")    # ang
# out_prob, score, index, text_lab = classifier.classify_file("audio/joyfully.wav")  #neu
# out_prob, score, index, text_lab = classifier.classify_file("speechbrain/emotion-recognition-wav2vec2-IEMOCAP/anger.wav")


print(text_lab)