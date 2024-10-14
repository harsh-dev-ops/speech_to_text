from flask_restful import fields, reqparse
from pydantic import BaseModel


VoiceRequestSchema = {
    "base64String": fields.String,
    "speechToText": fields.Integer,
    "enableNoiseReduction": fields.Boolean,
    "noiseReducer": fields.Integer
}

voice_request_parser = reqparse.RequestParser()
voice_request_parser.add_argument(
    'base64String',
    type=str,
    required=True,
    help="base64String needed in string format."
)
voice_request_parser.add_argument(
    'speechToText',
    type=int,
    required=False,
    help="speechToText needed in int format."
)
voice_request_parser.add_argument(
    'enableNoiseReduction',
    type=bool,
    required=False,
    help="enableNoiseReduction needed in bool format."
)
voice_request_parser.add_argument(
    'noiseReducer',
    type=int,
    required=False,
    help="noiseReducer needed in int format."
)


class VoiceRequest(BaseModel):
    base64String: str | None = None
    speechToText: int = 2
    enableNoiseReduction: bool = True
    noiseReducer: int = 1
