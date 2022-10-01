import os

import requests
from google.cloud import speech


class ASR:

    def __init__(self, name):
        self.name = name
        self.num_candidates = 3

    def run(self, audio_path):
        with open(audio_path, 'rb') as f:
            content = f.read()
            return self.recognise(audio_content=content)

    def recognise(self, audio_content):
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

    def run(self, audio_path):
        with open(audio_path, 'rb') as f:
            content = f.read()
            return self.recognise(audio_content=content)

    def recognise(self, audio_content):
        # LINEAR16 = Uncompressed 16-bit signed little-endian samples (Linear PCM).
        # pcm_s16le = PCM signed 16-bit little-endian
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

    def recognise(self, audio_content):
        response = requests.post(self.api_endpoint, files={'audio': audio_content},
                                 data={'num_candidates': self.num_candidates})

        return [prediction['transcript'].lower().strip() for prediction in response.json()]
