Send recording of 30 sec to getr a voice print (hash) 
Then send it to API endpoint "identify" - diarization du fichier + au lieu d'avoir speaker 1, il dira gradium

overlap de base avec la diarization 

BONUS:
another endpoint, exclusive diarization = permet d'avoir les zones de non-overlap, useful to make the diff with the actual percentage of non overlap

toolkit open source
pyannote.core = pour manipuler des objets; petit conversion pour avoir des Annotation and smth else objects
pyannote.metrics = DER, etc