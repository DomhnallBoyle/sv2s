import os
import time

import requests
import whisper
from google.cloud import speech


class ASR:

    def __init__(self, name):
        self.name = name
        self.num_candidates = 3
        self.num_samples = 0
        self.total_time_taken = 0

    @property
    def average_time(self): 
        return round(self.total_time_taken / self.num_samples, 1)

    def run(self, audio_path):
        self.num_samples += 1

        start_time = time.time()
        results = self.recognise(audio_path=audio_path)
        self.total_time_taken += (time.time() - start_time)

        return results

    def recognise(self, audio_path):
        raise NotImplementedError


class GoogleASR(ASR):

    def __init__(self, gcloud_credentials_path, phrases, model, language_code, sample_rate):
        super().__init__(name='Google')
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcloud_credentials_path
        print(f'Using phrases for ASR speech context:\n{phrases}')
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
            max_alternatives=self.num_candidates,
            use_enhanced=True,
            model=model,
            speech_contexts=[speech.SpeechContext(phrases=[p for p in phrases if len(p) <= 100])]  # needs <= 100 chars
        )

    def recognise(self, audio_path):
        # LINEAR16 = Uncompressed 16-bit signed little-endian samples (Linear PCM).
        # pcm_s16le = PCM signed 16-bit little-endian
        with open(audio_path, 'rb') as f:
            audio_content = f.read()

        response = self.client.recognize(
            config=self.config,
            audio=speech.RecognitionAudio(content=audio_content)
        )

        return [alternative.transcript.lower().strip()
                for result in response.results
                for alternative in result.alternatives]


class DeepSpeechASR(ASR):

    def __init__(self, host):
        super().__init__(name='DeepSpeech')
        self.api_endpoint = f'http://{host}/transcribe'

    def recognise(self, audio_path):
        with open(audio_path, 'rb') as f:
            audio_content = f.read()

        response = requests.post(self.api_endpoint, files={'audio': audio_content},
                                 data={'num_candidates': self.num_candidates})

        return [prediction['transcript'].lower().strip() for prediction in response.json()]


class WhisperASR(ASR): 
    
    def __init__(self, model, phrases=[]):
        super().__init__(name='Whisper')
        self.model = whisper.load_model(model)
        self.vocab = list(set([w for phrase in phrases for w in phrase.split(' ')]))  # vocab is the unique words 
        
    def recognise(self, audio_path): 
        results = self.model.transcribe(audio_path, initial_prompt=' '.join(self.vocab))

        return [results['text'].lower().strip().replace('.', '')]
