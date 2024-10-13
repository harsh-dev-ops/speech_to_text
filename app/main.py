from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse, marshal, marshal_with

from schema import VoiceRequestSchema, voice_request_parser
from voice.algorithms import ai_algorithms

app = Flask(__name__)
api = Api(app)


class SpeechToText(Resource):
    def get(self):
        algorithms = ai_algorithms()
        return jsonify(algorithms)

    @marshal_with(VoiceRequestSchema)
    def post(self):
        data = voice_request_parser.parse_args()
        return data, 200


class FileUpload(Resource):
    def post(self):
        # Check if the file was uploaded
        if 'file' not in request.files:
            return {'message': 'No file part in the request'}, 400

        file = request.files['file']

        # Check if the file has a filename
        if file.filename == '':
            return {'message': 'No selected file'}, 400

        # Save the file
        try:
            file.save('uploads/' + file.filename)
            return {'message': 'File uploaded successfully'}, 201
        except Exception as e:
            return {'message': 'Failed to upload file: ' + str(e)}, 500


api.add_resource(FileUpload, '/upload')
api.add_resource(SpeechToText, '/')

if __name__ == "__main__":
    app.run(port=8000, debug=True)
