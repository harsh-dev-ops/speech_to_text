import os
from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse, marshal, marshal_with

from schema import VoiceRequestSchema, voice_request_parser, VoiceRequest
from voice.algorithms import ai_algorithms
from services import SpeechToTextServiceCreator, FileService

app = Flask(__name__)
api = Api(app)


class SpeechToTextBase64Resource(Resource):
    def get(self):
        algorithms = ai_algorithms()
        return jsonify(algorithms)

    # @marshal_with(VoiceRequestSchema)
    def post(self):
        path = None
        # if 'file' not in request.files:
        #     return {'message': 'No file part in the request'}, 400

        # file = request.files['file']
        # file_service = FileService()
        # path = file_service._save(file)

        data = voice_request_parser.parse_args()

        data = VoiceRequest(**data)

        speech_to_text_service = SpeechToTextServiceCreator.speech_to_text(
            data, base64String=data.base64String
        )

        pred_text = speech_to_text_service.transcribe()

        return {
            "text": pred_text
        }, 200


class SpeechToTextFileResource(Resource):
    def get(self):
        algorithms = ai_algorithms()
        return jsonify(algorithms)

    # @marshal_with(VoiceRequestSchema)
    def post(self):
        if 'file' not in request.files:
            return {'message': 'No file part in the request'}, 400

        file = request.files['file']
        file_service = FileService()
        file_service.upload(file)

        path = os.path.join(file_service.save_dir, file.filename)

        data = {
            "base64String": None,
            "enableNoiseReduction": request.form.get("enableNoiseReduction", True),
            "noiseReducer": request.form.get("noiseReducer", 1),
            "speechToText": request.form.get("speechToText", 2),
        }

        data = VoiceRequest(**data)

        speech_to_text_service = SpeechToTextServiceCreator.speech_to_text(
            data, path=path
        )

        pred_text = speech_to_text_service.transcribe()

        return {
            "text": pred_text
        }, 200


class FileUploadResource(Resource):
    def post(self):

        if 'file' not in request.files:
            return {'message': 'No file part in the request'}, 400

        file = request.files['file']
        file_service = FileService()
        return file_service.upload(file)


class FileToBase64Resource(Resource):
    def post(self):
        if 'file' not in request.files:
            return {'message': 'No file part in the request'}, 400
        file = request.files['file']

        file_service = FileService()
        base64String = file_service.base64_string(file)
        return jsonify({"base64String": base64String})


api.add_resource(FileUploadResource, '/media')
api.add_resource(FileToBase64Resource, '/fileTobase64')
api.add_resource(SpeechToTextBase64Resource, '/speechToText/base64')
api.add_resource(SpeechToTextFileResource, '/speechToText/media')


if __name__ == "__main__":
    app.run(port=8000, debug=True)
