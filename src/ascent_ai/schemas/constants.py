# # Timeout for the medical coder api calls
MEDICAL_CODER_TIMEOUT = 1000


class CodingType:
    STANDARD_CODING = "Standard"
    SOURCE_CODING = "Source"

    @classmethod
    def values(cls):
        return [cls.STANDARD_CODING, cls.SOURCE_CODING]
