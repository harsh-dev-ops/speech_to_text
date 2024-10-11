from flask_restful import fields, reqparse


VoiceRequestSchema = {
    "audio": fields.String
}

voice_request_parser = reqparse.RequestParser()
voice_request_parser.add_argument(
    'audio',
    type=str,
    required=True,
    help="Audio needed in base64 string format."
)
