import time
import jwt
import requests
# from dotenv import load_dotenv
import os

# load_dotenv()

service_account_id = "ajenc5g0tj4sn0nuegtc"
key_id = "ajec25g64b9unk6iubqc"
private_key = 'PLEASE DO NOT REMOVE THIS LINE! Yandex.Cloud SA Key ID <ajec25g64b9unk6iubqc>\n-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDUgpqIK4n0VZxs\nBGtAcD1pmASEF1V/lEIqn+l1WRl/rDgmOAEXuaHaVlt3OiSyuBCzx9cFwr4VbYSu\nqRmy4l1MpFCrUzL1UCp7AFwre8dYCWHjtyyCbKoFQb6Xr55Z3lmurInFcmtsxSlK\n/3eM4LQjwg6qAocU+662vlbxkQpxZhOmvUJz74GOqStJPXM1QehhAIJvkKyfMxta\n1f4dV1keD9wkr/SGFjpiy5PE6dmDXR/ZhcoF7Fsjxpx03IGFvuKt20HDdXaPKmzj\ndZA14l+AmSrpZaKeT1T/sQ1AYAisb5CbGXEs95VKVcmrRYFaUNh+x5TKwcyGn7gE\nHbU2tTGzAgMBAAECggEAAl0ljOg3E6G0KLtv2fqlpDwNqM59o/qpNcIrnaONFg97\nXGl5EaN9+mjFgbEC4X4MqYNKkATXinN6a1r0Lzo3YXfecdluEq5+mnmpjM5GrGMG\nVLsf4p+E0KQnk5Zfg45mnvvKWIjqSv5ydCQdV1LUjiVFdnyqtgAKtHuvieZcIGDG\nGfUuvzO99Y81qPIIkVjAEvCJZbURyLLKZ9zFVipCAzy3vedVqMw8mbZ0u1TAKQcE\n9d57IF+o47/IDhbZI6XMh9QPNkp0bb0f/PgIdO3KixzYB8BCFZr3EKrTJei2S28c\n05a7H7YNMvZQ7JOMjWOCLCn6jk7BlaOA/+tshuXPcQKBgQD4+4bM/Q+Zw+eMfuLE\ns7bTDd/7kAj/Aos4I4p/xH49+9T60U/C4lyUJ22sPUGC75IcYeMhyaz9Xp8h8ZcT\nXT3FN0XwPw0mYX8QMQmN808h3TyLW217aivYK52MtV56pef3adra5tC2KBL+cJ6f\nz9fz9q8knue68aS9pyZaGLhWKwKBgQDaf+tpvaNyv3Ww/yMEaH+ywYemvltCPJXx\nCIcfNB45KP8weJKQaGolXyp6ZF8fha7srkMkP3uso47KRdnkrZ4zA+VevQ/q+YCd\noTUT1OyfS8kVVmJDWEqVMB7N5Jo9jqOZpfyrMFFk7FoCwEere7OxBnZMQHYGNLrX\ncvDgzGYWmQKBgQC2SgFBp616WLH4bRW+Cg26rBfm6GeNvOEM8wh9zvDWlMAz+nc5\nKd26IrvrNNX39Uq2OPzAShW7U0GS6nw/ky6ca4FrCd6o0QzkX+Ks6QxwsLeZGBEq\nIGuFUzmAQXGwvjL9M6UmS4NXOjPd0bpxKwzi8yL73tOuTSjeKbiskhi0DQKBgQDO\ndseVMFvNJvtn135YQQJBgDvVJNSI30Tz8JH1u0K/0mxoedZMXE3IqIc0BboYyDKF\ndRj+nHoZpKddnOmK+z+chxbrEY2EBGUzDcxgw/cfgvYskmbhqgE3vbOt7FCO0ETo\n//6kKFERI4DeTqCqeoZORYPtA5BCxvqycOsKEDp7KQKBgQDEgKly9mdpClMr7mrr\nLtZ8FlazkxslY1a4NSEMevIQrv+nUfDeTNeReABuwGoc+RkY5AheS4BmncZvGS0Z\nSabuUbJi228mDCHVe9VPtywGPJjV3OidWF2hHR7ed1zN6M1VNxkQ2WSducew+ywr\ncXzBkDhERQHUvpVq8UJ5mrlLdQ==\n-----END PRIVATE KEY-----\n'

def generate_token():
    # Получаем IAM-токен
    now = int(time.time())
    payload = {
            'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
            'iss': service_account_id,
            'iat': now,
            'exp': now + 360}

    # Формирование JWT
    encoded_token = jwt.encode(
        payload,
        private_key,
        algorithm='PS256',
        headers={'kid': key_id})

    url = 'https://iam.api.cloud.yandex.net/iam/v1/tokens'
    x = requests.post(url,  
                      headers={'Content-Type': 'application/json'},
                      json = {'jwt': encoded_token}).json()
    token = x['iamToken']
    return token