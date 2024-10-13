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


class VoiceRequest(BaseModel):
    base64String: str | None = None
    speechToText: int
    enableNoiseReduction: bool = True
    noiseReducer: int | None = None
